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
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

log = logging.getLogger(__name__)

CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "car_ads")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
COL = _client.get_or_create_collection(CHROMA_COLLECTION)
store = Chroma(client=_client, collection_name=CHROMA_COLLECTION, embedding_function=embeddings)


class RetrieveAdsByBrandModelView(APIView):
    """
    Cross-type retrieval:
      - If seed is BUY  -> shortlist SELL ads of same brand/model, then range-aware scoring.
      - If seed is SELL -> shortlist SELL ads (or you can later expand to also show BUY ask matches).
    """

    # SELL numeric keys (candidate cleaning)
    NUMERIC_KEYS_SELL = ["year", "mileage", "insurance_mo", "price", "lat", "lon"]

    def _to_float_or_none(self, v):
        try:
            if v is None: return None
            s = str(v).strip()
            if s == "": return None
            return float(s)
        except Exception:
            return None

    def _clean_sell(self, meta: dict):
        # Keep the ad and normalize types; missing -> None (we'll mask in scoring)
        m = dict(meta)
        for k in self.NUMERIC_KEYS_SELL:
            m[k] = self._to_float_or_none(m.get(k))
        # color_id stays a normalized string; empty -> ""
        m["color_id"] = str(m.get("color_id", "")).strip().lower()
        return m

    def post(self, request):
        ser = BrandModelRetrieveSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        seed_id = ser.validated_data["id"]
        excl_seed = ser.validated_data["exclude_seed"]
        thresh = ser.validated_data["threshold"]

        # 1) fetch seed ad (meta + embedding)
        seed = COL.get(ids=[str(seed_id)], include=["metadatas", "embeddings"])
        if not seed["ids"]:
            return Response({"detail": "Seed ad not found"}, status=404)

        seed_meta = seed["metadatas"][0]
        seed_vec = seed["embeddings"][0]
        seed_type = seed_meta.get("type", "sell")
        brand_id = seed_meta.get("brand_id")
        model_id = seed_meta.get("model_id")

        # 2) shortlist via ANN (same brand & model, opposite type if BUY)
        target_type = "sell" if seed_type == "buy" else "sell"  # currently return SELL candidates for both

        where = {"$and": [{"brand_id": brand_id}, {"model_id": model_id}, {"type": target_type}]}

        hits = COL.query(
            query_embeddings=[seed_vec],
            n_results=TOP_K,
            where=where,
            include=["metadatas", "embeddings", "distances"],
        )

        cand_ids = hits["ids"][0] if hits.get("ids") else []
        cand_meta = hits["metadatas"][0] if hits.get("metadatas") else []
        cand_vecs = hits["embeddings"][0] if hits.get("embeddings") else []

        rows = []
        for i, meta, vec in zip(cand_ids, cand_meta, cand_vecs):
            if excl_seed and str(i) == str(seed_id):
                continue
            # we currently only score SELL candidates (single values)
            clean = self._clean_sell(meta)
            rows.append((i, clean, vec))

        if not rows:
            return Response({"ads": []}, status=200)

        ids, metas, vecs = map(list, zip(*rows))

        # 3) second-pass Python ranking with missing-field masking
        if seed_type == "buy":
            scores = compliance_scores_mixed(seed_meta, metas, seed_vec, vecs)
        else:
            scores = compliance_scores(seed_meta, metas, seed_vec, vecs)

        ads = [
            {"id": i,
             "score_pct": round(float(s) * 100, 1),
             "metadata": meta}
            for i, s, meta in zip(ids, scores, metas)
            if s >= thresh
        ]
        ads.sort(key=lambda x: x["score_pct"], reverse=True)

        # Debug (optional)
        log.warning("Seed meta=%s", pprint.pformat(seed_meta))
        log.warning("First cand=%s", pprint.pformat(metas[0]) if metas else "N/A")
        log.warning("len(seed_vec)=%d  len(cand_vec0)=%d",
                    len(seed_vec), len(vecs[0]) if vecs else -1)

        return Response({"ads": ads}, status=200)


RABBIT_HOST = settings.RABBIT_HOST
RABBIT_PORT = settings.RABBIT_PORT
RABBIT_USER = settings.RABBIT_USER
RABBIT_PASS = settings.RABBIT_PASS
QUEUE_NAME = "car_ads"


def _open_channel():
    creds = pika.PlainCredentials(RABBIT_USER, RABBIT_PASS)
    params = pika.ConnectionParameters(host=RABBIT_HOST,
                                       port=RABBIT_PORT,
                                       credentials=creds,
                                       blocked_connection_timeout=2)
    conn = pika.BlockingConnection(params)
    return conn, conn.channel()


@api_view(["GET"])
def queue_peek(request):
    """
    GET /api/v1/compliance/queue-peek/?max=5

    Returns up to `max` un-acked messages (default 5) **without** removing
    them from the queue (requeued).
    """
    max_n = int(request.GET.get("max", 5))
    msgs = []

    try:
        conn, ch = _open_channel()

        # passive declare to fetch message_count
        q = ch.queue_declare(queue=QUEUE_NAME, passive=True)
        total = q.method.message_count

        for _ in range(max_n):
            method, props, body = ch.basic_get(queue=QUEUE_NAME, auto_ack=False)
            if method is None:  # queue empty
                break
            # body is bytes → json
            try:
                payload = json.loads(body)
            except Exception:
                payload = body.decode(errors="ignore")
            msgs.append(payload)
            # requeue so worker will still get it
            ch.basic_nack(method.delivery_tag, requeue=True)

        conn.close()
        return Response({"total_in_queue": total, "peek": msgs}, status=200)

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
