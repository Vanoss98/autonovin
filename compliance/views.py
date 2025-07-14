from rest_framework.views import APIView
from rest_framework.response import Response
import os, chromadb
from .serializers import BrandModelRetrieveSerializer
from .utils import compliance_scores
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "car_ads")

# 1) embeddings helper
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)  # or omit dimensions if 1536

# 2) low-level client → get (or create) the collection *once*
_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
COL = _client.get_or_create_collection(CHROMA_COLLECTION)

# 3) LangChain wrapper (optional – for add_texts, similarity_search, etc.)
store = Chroma(
    client=_client,
    collection_name=CHROMA_COLLECTION,
    embedding_function=embeddings,
)

BATCH = 500
GEO_SIGMA = 50


class RetrieveAdsByBrandModelView(APIView):
    NUMERIC_KEYS = ["year", "mileage", "insurance_mo", "price", "lat", "lon"]

    def _clean(self, meta):
        try:
            m = dict(meta)
            for k in self.NUMERIC_KEYS:
                m[k] = float(m[k])
            m["color"] = str(m["color"]).strip().lower()
            return m if m["color"] else None
        except Exception:
            return None

    def post(self, request):
        ser = BrandModelRetrieveSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        seed_id, excl = ser.validated_data["id"], ser.validated_data["exclude_seed"]

        seed = COL.get(ids=[str(seed_id)], include=["metadatas", "embeddings"])
        if not seed["ids"]:
            return Response({"detail": "Seed ad not found"}, status=404)

        seed_meta = seed["metadatas"][0];
        seed_vec = seed["embeddings"][0]
        b, m = seed_meta["brand"], seed_meta["model"]

        docs = COL.get(where={"$and": [{"brand": b}, {"model": m}]},
                       include=["metadatas", "embeddings"])
        rows = [
            (int(i), self._clean(meta), emb)
            for i, meta, emb in zip(docs["ids"], docs["metadatas"], docs["embeddings"])
            if (not excl or int(i) != seed_id) and meta is not None
        ]
        if not rows:
            return Response({"ads": []}, status=200)

        ids, metas, vecs = map(list, zip(*rows))
        import logging, pprint
        log = logging.getLogger(__name__)
        log.warning("Seed meta=%s", pprint.pformat(seed_meta))
        log.warning("First cand meta=%s", pprint.pformat(metas[0]))
        log.warning("len(seed_vec)=%d  len(cand_vec[0])=%d",
                    len(seed_vec), len(vecs[0]))
        scores = compliance_scores(seed_meta, metas, seed_vec, vecs)

        thresh = ser.validated_data["threshold"]

        ads = [
            {
                "id": int(i),
                "score_pct": round(float(s) * 100, 1),
                "metadata": meta,
            }
            for i, s, meta in zip(ids, scores, metas)
            if s >= thresh  # ← keep only scores ≥ threshold
        ]

        ads.sort(key=lambda x: x["score_pct"], reverse=True)
        return Response({"ads": ads}, status=200)


import pika, json, os
from rest_framework.decorators import api_view
from rest_framework.response import Response

RABBIT_HOST = os.getenv("RABBIT_HOST", "192.168.1.165")
RABBIT_PORT = int(os.getenv("RABBIT_PORT", 5672))
RABBIT_USER = os.getenv("RABBIT_USER", "guest1")
RABBIT_PASS = os.getenv("RABBIT_PASS", "guest1")
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
    msgs  = []

    try:
        conn, ch = _open_channel()

        # passive declare to fetch message_count
        q = ch.queue_declare(queue=QUEUE_NAME, passive=True)
        total = q.method.message_count

        for _ in range(max_n):
            method, props, body = ch.basic_get(queue=QUEUE_NAME, auto_ack=False)
            if method is None:      # queue empty
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
