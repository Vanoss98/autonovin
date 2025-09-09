# compliance/tasks.py
import json
import os
import chromadb
from celery import shared_task
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "car_ads")

# --- embeddings ----------------------------------------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024,
    api_key=OPENAI_KEY,
)

# --- Chroma connection ---------------------------------------------------
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
COL = client.get_or_create_collection(CHROMA_COLLECTION)


# -------------------------- text encoders --------------------------------
def ad_to_text_sell(ad_meta: dict) -> str:
    """Deterministic sentence for SELL ads (single values)."""
    # brand_id/model_id/color_id because IDs are what we store/filter on
    return (
        f"SELL brand:{ad_meta.get('brand_id')} model:{ad_meta.get('model_id')} "
        f"color:{ad_meta.get('color_id')} "
        f"year:{ad_meta.get('year')} km:{ad_meta.get('mileage')} "
        f"price:{ad_meta.get('price')} insurance_mo:{ad_meta.get('insurance_mo')} "
        f"@({ad_meta.get('lat')},{ad_meta.get('lon')})"
    )


def ad_to_text_buy(ad_meta: dict) -> str:
    """Deterministic sentence for BUY ads (ranges)."""
    return (
        f"BUY brand:{ad_meta.get('brand_id')} model:{ad_meta.get('model_id')} "
        f"color:{ad_meta.get('color_id')} "
        f"year:{ad_meta.get('from_year')}-{ad_meta.get('to_year')} "
        f"km:{ad_meta.get('from_km')}-{ad_meta.get('to_km')} "
        f"price:{ad_meta.get('from_price')}-{ad_meta.get('to_price')} "
        f"insurance_mo:{ad_meta.get('from_insurance_mo')}-{ad_meta.get('to_insurance_mo')} "
        f"@({ad_meta.get('lat')},{ad_meta.get('lon')})"
    )


# -------------------------- payload parsers -------------------------------
def _parse_buy_payload(message: dict) -> dict:
    """
    Keep ONLY the fields you asked for, and normalize names.
    Input is body["message"] from RabbitMQ for BUY.
    """
    return {
        "type": "buy",
        "brand_id": int(message["brandId"]),
        "model_id": int(message["cartipId"]),
        "color_id": str(message["colorId"]),
        "lat": float(message["customerAdvertisementLat"]),
        "lon": float(message["customerAdvertisementLon"]),
        "from_price": float(message["fromPrice"]),
        "to_price": float(message["toPrice"]),
        "from_km": float(message["fromKiloometer"]),
        "to_km": float(message["toKiloometer"]),
        "from_year": float(message["fromYearOfProduction"]),
        "to_year": float(message["toYearOfProduction"]),
        "from_insurance_mo": float(message["fromInsuranceDeadLine"]),
        "to_insurance_mo": float(message["toInsuranceDeadLine"]),
    }


def _parse_sell_payload(message: dict) -> dict:
    """
    SELL parser (single values). If your SELL source already matches the older schema,
    map to the normalized keys below. Adjust mapping if your SELL MQ message differs.
    """
    # Try V1 (old schema)
    if all(k in message for k in ("year", "brand", "model", "color", "mileage", "price", "insurance_mo", "lat", "lon")):
        return {
            "type": "sell",
            "brand_id": int(message.get("brand_id") or message["brand"]),
            "model_id": int(message.get("model_id") or message["model"]),
            "color_id": str(message.get("color_id") or message["color"]),
            "year": float(message["year"]),
            "mileage": float(message["mileage"]),
            "price": float(message["price"]),
            "insurance_mo": float(message["insurance_mo"]),
            "lat": float(message["lat"]),
            "lon": float(message["lon"]),
        }

    # Try V2 (ID-based schema)
    # Adjust these keys to your SELL producer if different.
    return {
        "type": "sell",
        "brand_id": int(message["brandId"]),
        "model_id": int(message["cartipId"]),
        "color_id": str(message["colorId"]),
        "year": float(message["yearOfProduction"]),
        "mileage": float(message["kilometer"]),
        "price": float(message["price"]),
        "insurance_mo": float(message.get("insuranceDeadLineMonths", 0)),
        "lat": float(message["lat"]),
        "lon": float(message["lon"]),
    }


def _stable_buy_id(message: dict) -> str:
    """
    If upstream id is null, create a stable composite id.
    Prefer a provided id if present; otherwise compose one.
    """
    raw_id = message.get("id")
    if raw_id:
        return f"buy:{raw_id}"
    return f"buy:{message.get('brandId')}:{message.get('cartipId')}:{message.get('colorId')}:" \
           f"{message.get('fromPrice')}-{message.get('toPrice')}:" \
           f"{message.get('fromKiloometer')}-{message.get('toKiloometer')}:" \
           f"{message.get('fromYearOfProduction')}-{message.get('toYearOfProduction')}"


def _stable_sell_id(message: dict) -> str:
    raw_id = message.get("id")
    if raw_id:
        return f"sell:{raw_id}"
    # Compose something stable from brand/model/price/km/year if no id
    return f"sell:{message.get('brandId') or message.get('brand')}:" \
           f"{message.get('cartipId') or message.get('model')}:" \
           f"{message.get('price')}:{message.get('kilometer') or message.get('mileage')}:" \
           f"{message.get('yearOfProduction') or message.get('year')}"


# ---------------------------- Celery task ---------------------------------
@shared_task(bind=True, name="compliance.upsert_ad_vector")
def upsert_ad_vector(self, ad_json: str):
    """
    Accepts the FULL MQ envelope as JSON (the thing you dumped with json.dumps(body)).
    Detects type (buy/sell) via messageType and upserts only the normalized meta + embedding.
    """
    payload = json.loads(ad_json)
    msg_types = payload.get("messageType", [])
    message = payload.get("message", {}) or {}

    if any("CustomerAdvertiseBuyEto" in t for t in msg_types):
        meta = _parse_buy_payload(message)
        text = ad_to_text_buy(meta)
        vec = embeddings.embed_query(text)
        ad_id = _stable_buy_id(message)
        COL.upsert(ids=[ad_id], embeddings=[vec], metadatas=[meta])
        return {"id": ad_id, "type": "buy", "status": "stored"}

    # Default: treat as SELL (adjust if you have a concrete sell type name)
    meta = _parse_sell_payload(message)
    text = ad_to_text_sell(meta)
    vec = embeddings.embed_query(text)
    ad_id = _stable_sell_id(message)
    COL.upsert(ids=[ad_id], embeddings=[vec], metadatas=[meta])
    return {"id": ad_id, "type": "sell", "status": "stored"}
