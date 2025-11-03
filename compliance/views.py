import os
import chromadb
import logging
import pprint
import pika, json, os
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings
from rest_framework.views import APIView
from .serializers import BrandModelRetrieveSerializer
from .utils import (
    compliance_scores,
    compliance_scores_mixed,
    TOP_K,
)
from rest_framework import status
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

log = logging.getLogger(__name__)

CHROMA_HOST = settings.CHROMA_HOST
CHROMA_PORT = int(settings.CHROMA_PORT)
CHROMA_COLLECTION = "car_ads"
OPENAI_KEY = settings.OPENAI_API_KEY

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024, api_key=OPENAI_KEY)
_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
COL = _client.get_or_create_collection(CHROMA_COLLECTION)
store = Chroma(client=_client, collection_name=CHROMA_COLLECTION, embedding_function=embeddings)


class RetrieveAdsByBrandModelView(APIView):
    """
    Cross-type retrieval:
      - BUY seed  -> shortlist SELL ads of same brand/model, then range-aware scoring.
      - SELL seed -> shortlist SELL ads.
    Excludes candidates with the same national_code as the seed (user’s own ads).

    NEW: Requires 'nationalCode' in payload. We verify the seed ad belongs to that national code
    before proceeding.
    """

    NUMERIC_KEYS_SELL = ["year", "mileage", "insurance_mo", "price", "lat", "lon"]

    def _to_float_or_none(self, v):
        try:
            if v is None:
                return None
            s = str(v).strip()
            if s == "":
                return None
            return float(s)
        except Exception:
            return None

    def _clean_sell(self, meta: dict):
        m = dict(meta or {})
        for k in self.NUMERIC_KEYS_SELL:
            m[k] = self._to_float_or_none(m.get(k))
        # keep color id as normalized string
        m["color_id"] = str(m.get("color_id", "")).strip().lower()
        # pass-through extra fields (helpful for UI)
        for k in ("color_name", "color_name_en", "national_code"):
            if k in meta:
                m[k] = meta.get(k)
        return m

    @staticmethod
    def _norm_nc(x):
        # normalize nationalCode for comparison; tweak if you want digits-only
        return (str(x or "")).strip()

    def post(self, request):
        ser = BrandModelRetrieveSerializer(data=request.data)
        ser.is_valid(raise_exception=True)

        seed_id   = ser.validated_data["id"]
        excl_seed = ser.validated_data["exclude_seed"]
        thresh    = ser.validated_data["threshold"]
        req_nc    = self._norm_nc(ser.validated_data["nationalCode"])

        # 1) fetch seed ad
        seed = COL.get(ids=[str(seed_id)], include=["metadatas", "embeddings"])
        if not seed.get("ids"):
            return Response({"detail": "Seed ad not found"}, status=status.HTTP_404_NOT_FOUND)

        seed_meta = seed["metadatas"][0] or {}
        seed_vec  = seed["embeddings"][0]
        seed_type = (seed_meta.get("type") or "sell").lower()
        seed_nc   = self._norm_nc(seed_meta.get("national_code"))

        # 1.a) nationalCode ownership check
        if not seed_nc:
            return Response(
                {"detail": "Seed ad has no national_code recorded; cannot verify ownership.",
                 "seed_id": seed_id},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if seed_nc != req_nc:
            return Response(
                {
                    "detail": "Seed ad does not belong to the provided nationalCode.",
                    "seed_id": seed_id,
                    "provided_nationalCode": req_nc,
                    "seed_national_code": seed_nc,
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        brand_id = seed_meta.get("brand_id")
        model_id = seed_meta.get("model_id")

        # 2) shortlist via ANN (same brand & model, SELL candidates)
        where = {"$and": [{"brand_id": brand_id}, {"model_id": model_id}, {"type": "sell"}]}

        hits = COL.query(
            query_embeddings=[seed_vec],
            n_results=TOP_K,
            where=where,
            include=["metadatas", "embeddings", "distances"],  # do NOT put "ids" here
        )

        cand_ids  = (hits.get("ids") or [[]])[0]
        cand_meta = (hits.get("metadatas") or [[]])[0]
        cand_vecs = (hits.get("embeddings") or [[]])[0]

        rows = []
        for i, meta, vec in zip(cand_ids, cand_meta, cand_vecs):
            if excl_seed and str(i) == str(seed_id):
                continue
            # exclude same national_code (user’s own ads)
            cand_nc = self._norm_nc((meta or {}).get("national_code"))
            if seed_nc and cand_nc and cand_nc == seed_nc:
                continue
            clean = self._clean_sell(meta)
            rows.append((i, clean, vec))

        if not rows:
            return Response({"ads": []}, status=status.HTTP_200_OK)

        ids, metas, vecs = map(list, zip(*rows))

        # 3) Python ranking (BUY seed uses range-aware scorer)
        if seed_type == "buy":
            scores = compliance_scores_mixed(seed_meta, metas, seed_vec, vecs)
        else:
            scores = compliance_scores(seed_meta, metas, seed_vec, vecs)

        ads = [
            {"id": i, "score_pct": round(float(s) * 100, 1), "metadata": meta}
            for i, s, meta in zip(ids, scores, metas)
            if s >= thresh
        ]
        ads.sort(key=lambda x: x["score_pct"], reverse=True)

        log.warning("Seed meta=%s", seed_meta)
        return Response({"ads": ads}, status=status.HTTP_200_OK)

RABBIT_HOST = settings.RABBIT_HOST
RABBIT_PORT = settings.RABBIT_PORT
RABBIT_USER = settings.RABBIT_USER
RABBIT_PASS = settings.RABBIT_PASS
QUEUE_NAME = "OrderManagement.Application.Contracts.OrderManagement.Dtos.CustomerAdvertisement.sell.Compliance.etowithque:CustomerAdvertiseSellComplianceEto_skipped"


def _open_channel():
    creds = pika.PlainCredentials(settings.RABBIT_USER, settings.RABBIT_PASS)
    params = pika.ConnectionParameters(
        host=settings.RABBIT_HOST,
        port=int(settings.RABBIT_PORT),
        virtual_host=getattr(settings, "RABBIT_VHOST", "/"),
        credentials=creds,
        blocked_connection_timeout=2,
    )
    conn = pika.BlockingConnection(params)
    return conn, conn.channel()


@api_view(["GET"])
def queue_peek(request):
    """
    GET /api/v1/compliance/queue-peek/?max=5&q=buy|sell
    Peeks up to `max` messages (requeues them). By default uses BUY queue.
    """
    max_n = int(request.GET.get("max", 5))
    qsel = (request.GET.get("q") or "buy").strip().lower()
    qname = settings.COMPLIANCE_QUEUE_BUY if qsel == "buy" else settings.COMPLIANCE_QUEUE_SELL

    msgs = []
    try:
        conn, ch = _open_channel()

        # passive declare to fetch message_count
        q = ch.queue_declare(queue=qname, passive=True)
        total = q.method.message_count

        for _ in range(max_n):
            method, props, body = ch.basic_get(queue=qname, auto_ack=False)
            if method is None:
                break
            try:
                payload = json.loads(body)
            except Exception:
                payload = body.decode(errors="ignore")

            # if you *only* want the message core uncomment the next line:
            # payload = (payload.get("message") if isinstance(payload, dict) else payload) or payload

            msgs.append(payload)
            ch.basic_nack(method.delivery_tag, requeue=True)

        conn.close()
        return Response({"queue": qname, "total_in_queue": total, "peek": msgs}, status=200)

    except Exception as exc:
        return Response({"detail": str(exc)}, status=503)


@api_view(["GET"])
def chroma_dump(request):
    """
    GET /api/v1/compliance/chroma-dump/?limit=50&offset=0&type=sell&brand_id=58&model_id=177&embeddings=0

    Lists items currently stored in the Chroma collection.
    - Pagination: limit, offset
    - Filters (optional): type ('sell'|'buy'), brand_id (int), model_id (int)
    - embeddings: 0/1 (default 0 = omit embeddings)
    """
    try:
        limit = int(request.GET.get("limit", 50))
        offset = int(request.GET.get("offset", 0))
        want_emb = request.GET.get("embeddings", "0") in ("1", "true", "True")

        # Build optional `where` filter
        filters = []
        t = request.GET.get("type")
        if t in ("sell", "buy"):
            filters.append({"type": t})

        def _maybe_int(name):
            val = request.GET.get(name)
            if val is not None and str(val).strip() != "":
                try:
                    return int(val)
                except Exception:
                    pass
            return None

        brand_id = _maybe_int("brand_id")
        model_id = _maybe_int("model_id")
        if brand_id is not None:
            filters.append({"brand_id": brand_id})
        if model_id is not None:
            filters.append({"model_id": model_id})

        where = {"$and": filters} if filters else None

        include = ["metadatas"]
        if want_emb:
            include.append("embeddings")

        # Page through the collection
        page = COL.get(where=where, limit=limit, offset=offset, include=include)
        total = COL.count()

        items = []
        for i, mid in enumerate(page.get("ids", [])):
            md = (page.get("metadatas") or [None] * 1)[i]
            emb = (page.get("embeddings") or [None] * 1)[i] if want_emb else None
            items.append({
                "id": mid,
                "type": (md or {}).get("type"),
                "brand_id": (md or {}).get("brand_id"),
                "model_id": (md or {}).get("model_id"),
                "metadata": md,
                **({"embedding_dim": len(emb)} if want_emb and emb is not None else {})
            })

        return Response({
            "collection": CHROMA_COLLECTION,
            "total": total,
            "returned": len(items),
            "limit": limit,
            "offset": offset,
            "filters": {"type": t, "brand_id": brand_id, "model_id": model_id},
            "items": items,
        }, status=200)

    except Exception as exc:
        return Response({"detail": str(exc)}, status=500)



# ---- Sample seeder -------------------------------------------------------
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from django.conf import settings
import time

def _ad_to_text_buy(m):
    return (f"BUY brand:{m.get('brand_id')} model:{m.get('model_id')} "
            f"color_id:{m.get('color_id')} color_en:{m.get('color_name_en')} "
            f"year:{m.get('from_year')}-{m.get('to_year')} "
            f"km:{m.get('from_km')}-{m.get('to_km')} "
            f"price:{m.get('from_price')}-{m.get('to_price')} "
            f"insurance_mo:{m.get('from_insurance_mo')}-{m.get('to_insurance_mo')} "
            f"@({m.get('lat')},{m.get('lon')})")

def _ad_to_text_sell(m):
    return (f"SELL brand:{m.get('brand_id')} model:{m.get('model_id')} "
            f"color_id:{m.get('color_id')} color_en:{m.get('color_name_en')} "
            f"year:{m.get('year')} km:{m.get('mileage')} "
            f"price:{m.get('price')} insurance_mo:{m.get('insurance_mo')} "
            f"@({m.get('lat')},{m.get('lon')})")

@api_view(["POST"])
@permission_classes([AllowAny])
def seed_sample_ads(request):
    """
    POST /api/v1/compliance/seed-sample-ads/

    Wipes the current Chroma collection and inserts 10 sample ads:
      - 5 BUY ads (brand_id=405, model_id=501, color_id '29' pearl white)
      - 5 SELL ads (brand_id=405, model_id=501, color_id '23' black)
    All with distinct national_code values for exclusion testing.
    """
    # 1) reset collection hard (delete + recreate)
    global COL
    coll_name = CHROMA_COLLECTION
    try:
        _client.delete_collection(coll_name)
    except Exception:
        pass
    COL = _client.get_or_create_collection(coll_name)

    # 2) define samples (same brand/model so your shortlist hits)
    buy_nc = ["1111111111","2222222222","3333333333","4444444444","5555555555"]
    sell_nc= ["6666666666","7777777777","8888888888","9999999999","0000000000"]

    # BUY templates (range-based)
    buys = []
    for i, nc in enumerate(buy_nc, start=1):
        buys.append({
            "id": f"buy:TEST{i:02d}",
            "metadata": {
                "type": "buy",
                "brand_id": 405,
                "model_id": 501,
                "color_id": "29",
                "color_name": "سفيد صدفي",
                "color_name_en": "pearl white",
                "from_year": 2020.0,
                "to_year":   2025.0,
                "from_km": 1.0,
                "to_km":   100.0,
                "from_price": 10.0,
                "to_price":   100000.0,
                "from_insurance_mo": 1.0,
                "to_insurance_mo":   2.0,
                "lat": None,
                "lon": None,
                "national_code": nc,
            }
        })

    # SELL templates (point-values)
    sells = []
    for i, nc in enumerate(sell_nc, start=1):
        sells.append({
            "id": f"sell:TEST{i:02d}",
            "metadata": {
                "type": "sell",
                "brand_id": 405,      # NOTE: match brand to BUY so your where-filter works
                "model_id": 501,
                "color_id": "23",
                "color_name": "مشکی",
                "color_name_en": "black",
                "year": 2024.0,
                "mileage": 10.0,
                "price": 100000000.0,
                "insurance_mo": 0.0,
                "lat": None,
                "lon": None,
                "national_code": nc,
            }
        })

    # 3) build embeddings and upsert
    ids, metas, embs = [], [], []

    # BUY embeddings
    for row in buys:
        text = _ad_to_text_buy(row["metadata"])
        vec = embeddings.embed_query(text)
        ids.append(row["id"])
        metas.append(row["metadata"])
        embs.append(vec)

    # SELL embeddings
    for row in sells:
        text = _ad_to_text_sell(row["metadata"])
        vec = embeddings.embed_query(text)
        ids.append(row["id"])
        metas.append(row["metadata"])
        embs.append(vec)

    # upsert once (faster than per-row)
    COL.upsert(ids=ids, metadatas=metas, embeddings=embs)

    # small summary payload similar to your chroma_dump shape
    items = []
    for i, mid in enumerate(ids):
        md = metas[i]
        items.append({
            "id": mid,
            "type": md.get("type"),
            "brand_id": md.get("brand_id"),
            "model_id": md.get("model_id"),
            "metadata": md,
        })

    return Response({
        "collection": coll_name,
        "inserted": len(items),
        "items": items,
    }, status=201)
