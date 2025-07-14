import json, os, chromadb
from celery import shared_task
from django.conf import settings
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
# --- embeddings ----------------------------------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024, api_key=OPENAI_KEY)

# --- Chroma connection ---------------------------------------------------
client = chromadb.HttpClient(
    host=os.getenv("CHROMA_HOST", "chroma"),
    port=int(os.getenv("CHROMA_PORT", 8000)),
)
COL = client.get_or_create_collection(os.getenv("CHROMA_COLLECTION", "car_ads"))

# --- deterministic sentence (same used in your view) ---------------------
def ad_to_text(ad):
    return (
        f"{ad['year']} {ad['brand']} {ad['model']} — "
        f"{ad['color']}, {ad['mileage']} km, ${ad['price']}, "
        f"{ad['insurance_mo']} mo insurance"
    )

@shared_task(bind=True, name="compliance.upsert_ad_vector")
def upsert_ad_vector(self, ad_json: str):
    ad = json.loads(ad_json)
    vec = embeddings.embed_query(ad_to_text(ad))
    COL.upsert(ids=[str(ad["id"])], embeddings=[vec], metadatas=[ad])
    return {"id": ad["id"], "status": "stored"}
