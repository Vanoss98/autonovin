# compliance/tasks.py
# -*- coding: utf-8 -*-
import json
import time
import logging
import chromadb
from celery import shared_task
from django.conf import settings
from langchain_openai import OpenAIEmbeddings

print("[TASKS] importing compliance.tasks ...")
log = logging.getLogger(__name__)

def _mask_tail(s, keep=4):
    if not s: return "MISSING"
    s = str(s)
    return "*"*(len(s)-keep) + s[-keep:] if len(s)>keep else "*"*len(s)

CHROMA_HOST = getattr(settings, "CHROMA_HOST", "chroma")
CHROMA_PORT = int(getattr(settings, "CHROMA_PORT", 8000))
CHROMA_COLLECTION = getattr(settings, "CHROMA_COLLECTION", "car_ads")
print(f"[TASKS] Chroma target: {CHROMA_HOST}:{CHROMA_PORT}/{CHROMA_COLLECTION}")
print(f"[TASKS] OPENAI key present? {bool(getattr(settings,'OPENAI_API_KEY', ''))} tail={_mask_tail(getattr(settings,'OPENAI_API_KEY', None))}")

# ----- embeddings ---------------------------------------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024,
    api_key=getattr(settings, "OPENAI_API_KEY", None),
)
print("[TASKS] OpenAIEmbeddings created")

# ----- chroma -------------------------------------------------------------
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
COL = client.get_or_create_collection(CHROMA_COLLECTION)
print("[TASKS] Chroma collection ready")

# ---- color maps ----
COLOR_FA = { 20:"آبی",21:"زرد",22:"قرمز",23:"مشکی",24:"سفيد",25:"خاکستري",26:"قهوه اي",27:"نقره اي",
    28:"نوک مدادي",29:"سفيد صدفي",30:"سرمه اي",31:"بژ",32:"تيتانيوم",33:"يشمي",34:"سبز",
    35:"آلبالويي",36:"کربن بلک",37:"سربي",38:"دلفيني",39:"نقرآبي",40:"بادمجاني",41:"مسي",
    42:"کرم",43:"زيتوني",44:"طوسي",45:"اطلسي",46:"ذغالي",47:"نارنجي",48:"گيلاسي",
    49:"عنابي",50:"طلايي",51:"برنز",52:"بنفش",53:"زرشکي",54:"عدسي",55:"پوست پيازي",
    56:"موکا",57:"خاکي",58:"مارون",59:"اخرايي",60:"صورتي",61:"ياسي",62:"شتري",
    63:"امبربلک",64:"تارتوفو",66:"نامشخص", }
COLOR_EN = { 20:"blue",21:"yellow",22:"red",23:"black",24:"white",25:"gray",26:"brown",27:"silver",
    28:"gunmetal",29:"pearl white",30:"navy",31:"beige",32:"titanium",33:"jade",34:"green",
    35:"cherry",36:"carbon black",37:"lead gray",38:"dolphin gray",39:"silver-blue",
    40:"eggplant",41:"copper",42:"cream",43:"olive",44:"smoke gray",45:"atlas blue",
    46:"charcoal",47:"orange",48:"cherry red",49:"maroon",50:"gold",51:"bronze",52:"purple",
    53:"crimson",54:"hazel",55:"onion-skin",56:"mocha",57:"khaki",58:"maroon",59:"ochre",
    60:"pink",61:"lilac",62:"camel",63:"amber black",64:"tartufo",66:"unknown", }

def _normalize_color_fields(color_id_raw):
    try: cid_int = int(str(color_id_raw).strip())
    except Exception: cid_int = 66
    return str(cid_int), COLOR_FA.get(cid_int, "نامشخص"), COLOR_EN.get(cid_int, "unknown")

def _to_int(v):
    try: return int(v)
    except Exception: return None

def _to_float(v):
    try:
        if v is None or str(v).strip()=="":
            return None
        return float(v)
    except Exception:
        return None

def _to_str(v):
    try:
        s=str(v)
        return s if s.strip()!="" else None
    except Exception:
        return None

# ---------- STRICT crud mode mapping (0 insert, 2 update, 3 delete) ------
def _crud_mode(message: dict) -> str:
    raw = message.get("crudMode")
    if raw is None:
        raw = message.get("curdMode") or message.get("CRUDMode") or message.get("crudmode")
    print(f"[TASK] crudMode raw={raw!r}")

    parsed_int = None
    if isinstance(raw, (int, float)) or (isinstance(raw, str) and raw.isdigit()):
        try:
            parsed_int = int(raw)
        except Exception:
            parsed_int = None
    print(f"[TASK] crudMode parsed_int={parsed_int!r}")

    if parsed_int is not None:
        if parsed_int == 3: return "delete"
        if parsed_int == 2: return "update"
        if parsed_int == 0: return "insert"
        # any other number → default to insert (SAFE)
        return "insert"

    s = _to_str(raw)
    if s:
        s = s.lower()
        if s in ("delete", "remove"): return "delete"
        if s in ("update", "edit"):   return "update"
        if s in ("insert", "create"): return "insert"

    return "insert"

def _is_buy(payload: dict) -> bool:
    tlist = payload.get("messageType") or []
    dest = _to_str(payload.get("destinationAddress")) or ""
    joined = " | ".join(tlist + [dest]).lower()
    return (".buy." in joined) or ("customeradvertisebuy" in joined)

def ad_to_text_sell(m):
    return (f"SELL brand:{m.get('brand_id')} model:{m.get('model_id')} "
            f"color_id:{m.get('color_id')} color_en:{m.get('color_name_en')} "
            f"year:{m.get('year')} km:{m.get('mileage')} "
            f"price:{m.get('price')} insurance_mo:{m.get('insurance_mo')} "
            f"@({m.get('lat')},{m.get('lon')})")

def ad_to_text_buy(m):
    return (f"BUY brand:{m.get('brand_id')} model:{m.get('model_id')} "
            f"color_id:{m.get('color_id')} color_en:{m.get('color_name_en')} "
            f"year:{m.get('from_year')}-{m.get('to_year')} "
            f"km:{m.get('from_km')}-{m.get('to_km')} "
            f"price:{m.get('from_price')}-{m.get('to_price')} "
            f"insurance_mo:{m.get('from_insurance_mo')}-{m.get('to_insurance_mo')} "
            f"@({m.get('lat')},{m.get('lon')})")

def _parse_buy_payload(message: dict) -> dict:
    cid, cname_fa, cname_en = _normalize_color_fields(message.get("colorId"))
    return {
        "type":"buy",
        "national_code": _to_str(message.get("nationalCode")),
        "brand_id": _to_int(message.get("brandId")),
        "model_id": _to_int(message.get("cartipId")),
        "color_id": cid, "color_name": cname_fa, "color_name_en": cname_en,
        "lat": _to_float(message.get("customerAdvertisementLat")),
        "lon": _to_float(message.get("customerAdvertisementLon")),
        "from_price": _to_float(message.get("fromPrice")),
        "to_price": _to_float(message.get("toPrice")),
        "from_km": _to_float(message.get("fromKiloometer")),
        "to_km": _to_float(message.get("toKiloometer")),
        "from_year": _to_float(message.get("fromYearOfProduction")),
        "to_year": _to_float(message.get("toYearOfProduction")),
        "from_insurance_mo": _to_float(message.get("fromInsuranceDeadLine")),
        "to_insurance_mo": _to_float(message.get("toInsuranceDeadLine")),
    }

def _parse_sell_payload(message: dict) -> dict:
    cid, cname_fa, cname_en = _normalize_color_fields(message.get("colorId"))
    return {
        "type":"sell",
        "national_code": _to_str(message.get("nationalCode")),
        "brand_id": _to_int(message.get("brandId")),
        "model_id": _to_int(message.get("cartipId")),
        "color_id": cid, "color_name": cname_fa, "color_name_en": cname_en,
        "year": _to_float(message.get("yearOfProduction")),
        "mileage": _to_float(message.get("kiloometer")),
        "price": _to_float(message.get("price")),
        "insurance_mo": _to_float(message.get("insuranceDeadLine")),
        "lat": _to_float(message.get("customerAdvertisementLat")),
        "lon": _to_float(message.get("customerAdvertisementLon")),
    }

def _id_with_prefix(msg: dict, typ: str) -> str:
    rid = msg.get("id")
    if rid is not None:
        return f"{typ}:{rid}"
    if typ == "buy":
        return (f"buy:{msg.get('brandId')}:{msg.get('cartipId')}:{msg.get('colorId')}:"
                f"{msg.get('fromPrice')}-{msg.get('toPrice')}:"
                f"{msg.get('fromKiloometer')}-{msg.get('toKilometer') or msg.get('toKiloometer')}:"
                f"{msg.get('fromYearOfProduction')}-{msg.get('toYearOfProduction')}")
    return (f"sell:{msg.get('brandId')}:{msg.get('cartipId')}:"
            f"{msg.get('price')}:{msg.get('kilometer') or msg.get('kiloometer')}:"
            f"{msg.get('yearOfProduction') or msg.get('year')}")

# ---------------------------- Celery task ---------------------------------
@shared_task(bind=True, name="compliance.upsert_ad_vector")
def upsert_ad_vector(self, ad_json: str):
    print(f"[TASK] upsert_ad_vector called; payload type={type(ad_json)}")
    if isinstance(ad_json, str):
        print(f"[TASK] raw envelope (first 600 chars): {ad_json[:600]}{'...<trunc>' if len(ad_json)>600 else ''}")
    try:
        payload = json.loads(ad_json) if isinstance(ad_json, str) else ad_json
    except Exception as e:
        print(f"[TASK] JSON decode ERROR: {e!r}")
        return {"status":"error","stage":"decode","detail":str(e)}

    msg_types = payload.get("messageType") or []
    dest = payload.get("destinationAddress")
    print(f"[TASK] envelope: destination={dest} messageType={msg_types}")

    message = payload.get("message") or {}
    typ = "buy" if _is_buy(payload) else "sell"
    mode = _crud_mode(message)   # STRICT mapping with prints
    ad_id = _id_with_prefix(message, typ)
    print(f"[TASK] detected typ={typ} mode={mode} id={ad_id}")
    print(f"[TASK] message keys: {list(message.keys())[:20]}")

    # ---- DELETE (only when mode == 'delete') -----------------------------
    if mode == "delete":
        # delete only this type
        ids = [ad_id]
        # also try legacy fallback for this type (no id)
        ids.append(_id_with_prefix({**message, "id": None}, typ))
        # dedup while preserving order
        seen = set(); ids = [x for x in ids if not (x in seen or seen.add(x))]
        print(f"[TASK] delete attempting ids={ids}")
        try:
            COL.delete(ids=ids)
            print("[TASK] delete done ✓")
        except Exception as e:
            print(f"[TASK] delete ERROR: {e!r}")
            return {"status":"error","stage":"delete","detail":str(e)}
        return {"ids_attempted": ids, "status": "deleted"}

    # ---- INSERT / UPDATE → UPSERT ---------------------------------------
    meta = _parse_buy_payload(message) if typ=="buy" else _parse_sell_payload(message)
    print(f"[TASK] parsed meta preview: { {k: meta.get(k) for k in ('type','brand_id','model_id','price','year','from_price','to_price')} }")
    text = ad_to_text_buy(meta) if typ=="buy" else ad_to_text_sell(meta)
    print(f"[TASK] embedding text (first 300 chars): {text[:300]}{'...<trunc>' if len(text)>300 else ''}")

    # embed
    t0 = time.perf_counter()
    try:
        vec = embeddings.embed_query(text)
    except Exception as e:
        print(f"[TASK] EMBEDDING ERROR: {e!r}")
        return {"status":"error","stage":"embed","detail":str(e)}
    dt = (time.perf_counter()-t0)*1000
    print(f"[TASK] embedding ok; dim={len(vec) if hasattr(vec,'__len__') else '?'} time_ms={dt:.1f}")

    # upsert
    try:
        print(f"[TASK] upsert → collection={CHROMA_COLLECTION} id={ad_id}")
        COL.upsert(ids=[ad_id], embeddings=[vec], metadatas=[meta])
        print("[TASK] upsert done ✓")
    except Exception as e:
        print(f"[TASK] UPSERT ERROR: {e!r}")
        return {"status":"error","stage":"upsert","detail":str(e)}

    return {"id": ad_id, "type": typ, "status": "stored", "mode": mode}
