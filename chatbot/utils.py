# utils.py
import os, re, json, itertools, logging, unicodedata
from typing import Dict, List, Tuple, Any
from functools import lru_cache

from django.conf import settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# BM25: سعی می‌کنیم اگر bm25s نصب بود از آن (خیلی سریع‌تر) استفاده کنیم؛
# در غیر اینصورت به rank_bm25 برمی‌گردیم.
_USE_BM25S = False
try:
    from bm25s import BM25 as BM25S  # pip install bm25s
    _USE_BM25S = True
except Exception:
    from rank_bm25 import BM25Okapi  # pip install rank_bm25
    BM25S = None

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

# اگر جای دیگری از پروژه LLM استفاده شود، کش سطح-پردازش فعال باشد
set_llm_cache(InMemoryCache())

# ───────────────────────────────────────────────────────────────────────────────
# Embeddings (بدون تغییر مدل — فقط LRU برای embed_query)
# ───────────────────────────────────────────────────────────────────────────────
EMBED_MODEL = "text-embedding-3-large"
# توجه: تعیین dimensions در این مدل پشتیبانی می‌شود؛ اگر ingest شما بر 1024 ساخته شده، همان را حفظ کنید.
EMBED_DIM = 1024
EMBED = OpenAIEmbeddings(model=EMBED_MODEL, dimensions=EMBED_DIM)

@lru_cache(maxsize=512)
def _embed_cached(q: str) -> List[float]:
    return EMBED.embed_query(q)

# ───────────────────────────────────────────────────────────────────────────────
# Vector DB
# ───────────────────────────────────────────────────────────────────────────────
from ingestion.service import chroma_client
vectordb = Chroma(
    client=chroma_client,
    collection_name="car_spec",
    embedding_function=EMBED,
)

# ───────────────────────────────────────────────────────────────────────────────
# نرمالایزر سریع فارسی/انگلیسی و اعداد
# ───────────────────────────────────────────────────────────────────────────────
_TATWEEL_RE = re.compile(r"[\u0640\u200c\u200f\u2060]")
_MULTI_WS_RE = re.compile(r"\s+")
_FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_EN_DIGITS = "0123456789"
_EN2FA = str.maketrans(_EN_DIGITS, _FA_DIGITS)
_FA2EN = str.maketrans(_FA_DIGITS, _EN_DIGITS)

def _normalize(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("ي","ی").replace("ك","ک")
    s = _TATWEEL_RE.sub("", s)
    s = _MULTI_WS_RE.sub(" ", s.strip().lower())
    return s

def _norm_variants(s: str) -> List[str]:
    s = _normalize(s)
    return [s.translate(_FA2EN), s.translate(_EN2FA)]

def _tokenize(s: str) -> List[str]:
    s = _normalize(s)
    # ساده: برش بر اساس فاصله و علائم
    toks = re.split(r"[^\w\u0600-\u06FF\-\+\.%/]+", s)
    return [t for t in toks if t]

# ───────────────────────────────────────────────────────────────────────────────
# واژه‌نامه سادهٔ نام برند/مدل (قابل گسترش)
# اگر فایل data/car_aliases.json وجود داشته باشد، بارگذاری می‌شود.
# فرمت فایل:
# { "BMW X5": {"aliases": ["BMW X5","بی‌ام‌و X5","بی ام و X5"], "brand": "BMW"}, ... }
# ───────────────────────────────────────────────────────────────────────────────
_ALIAS_MAP: Dict[str, Dict[str, Any]] = {}

def _load_aliases_from_file() -> Dict[str, Dict[str, Any]]:
    base = os.path.dirname(__file__)
    candidate = os.path.join(base, "data", "car_aliases.json")
    if os.path.exists(candidate):
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}

def _bootstrap_default_aliases() -> Dict[str, Dict[str, Any]]:
    # حداقل برندهای رایج برای Normalize سریع (می‌توانید گسترش دهید)
    seed = {
        "BMW": {"aliases": ["BMW", "بی‌ام‌و", "بی ام و"], "brand": "BMW"},
        "Audi A3": {"aliases": ["Audi A3","آئودی A3","ائودی A3"], "brand": "Audi"},
        "BMW X5": {"aliases": ["BMW X5","بی‌ام‌و X5","بی ام و X5"], "brand": "BMW"},
        "Toyota Corolla": {"aliases": ["Toyota Corolla","تویوتا کرولا"], "brand": "Toyota"},
    }
    return seed

def _build_alias_index() -> Dict[str, Dict[str, Any]]:
    data = _load_aliases_from_file()
    if not data:
        data = _bootstrap_default_aliases()
    # نرمال‌سازی تمام aliasها به هر دو واریانت عددی
    idx: Dict[str, Dict[str, Any]] = {}
    for canon, meta in data.items():
        als = meta.get("aliases", []) + [canon]
        normed = set()
        for a in als:
            for v in _norm_variants(a):
                if v:
                    normed.add(v)
        idx[canon] = {"brand": meta.get("brand",""), "aliases": sorted(normed)}
    return idx

_ALIAS_MAP = _build_alias_index()

# ───────────────────────────────────────────────────────────────────────────────
# تحلیل سریع محلی (جایگزین analyze_query مبتنی بر LLM)
# ───────────────────────────────────────────────────────────────────────────────
def fast_analyze_query(user_query: str) -> Dict[str, Any]:
    """
    خروجی:
    {
      "models_hit": ["BMW X5", "Audi A3", ...],   # اگر پارس شدند
      "alias_hits": ["bmw x5","بی‌ام‌و x5", ...],  # واریانت‌های نرمال‌شده‌ای که مچ شدند
      "tokens": ["شتاب","0-100","hp","nm",...],    # توکن‌های کوئری
    }
    """
    q = _normalize(user_query)
    tokens = _tokenize(user_query)
    q_variants = set(_norm_variants(user_query))

    models_hit, alias_hits = [], []
    for canon, info in _ALIAS_MAP.items():
        for alias in info["aliases"]:
            # مچ نرم: وجود زیررشته در هر واریانت
            if any(alias in v for v in q_variants) or alias in q:
                models_hit.append(canon)
                alias_hits.extend(info["aliases"])
                break

    # حذف تکراری
    models_hit = list(dict.fromkeys(models_hit))
    alias_hits = list(dict.fromkeys(alias_hits))
    return {"models_hit": models_hit, "alias_hits": alias_hits, "tokens": tokens}

# ───────────────────────────────────────────────────────────────────────────────
# جست‌وجوی هیبریدی تک‌برداری (Embed یک‌باره روی خود پرسش کاربر)
# ───────────────────────────────────────────────────────────────────────────────
def _to_similarity_from_distance(d: float) -> float:
    # chroma در LangChain distance می‌دهد (کوچکتر=مشابه‌تر).
    # آن‌را به شباهت در (0,1] تبدیل می‌کنیم تا با BM25 قابل فیوژن باشد.
    return 1.0 / (1.0 + max(d, 0.0))

def fast_hybrid_search(
    user_query: str,
    parsed: Dict[str, Any],
    *,
    top_k: int = 20,
    alpha: float = 0.60,  # وزن embedding similarity
    beta: float  = 0.35,  # وزن BM25
    gamma: float = 0.05,  # بونس ظریف کلیدواژه/مدل
) -> Dict[str, List[Tuple]]:
    """
    خروجی سازگار با نسخهٔ قبل:
      { "<canon_or_all>": [ (id,title,content,score,images,model_id,url,source,page_id), ... ] }
    """
    q_norm = _normalize(user_query)
    vec = _embed_cached(q_norm)

    # اگر aliasهایی مچ شده‌اند، با where فیلتر می‌کنیم (سریع‌تر).
    where_meta = None
    aliases = parsed.get("alias_hits") or []
    if aliases:
        where_meta = {"car_model": {"$in": aliases}}

    # 1) ANN با فیلتر اختیاری متادیتا (سریع)
    # NOTE: LangChain Chroma از filter و where_document پشتیبانی می‌کند.
    doc_scores = vectordb.similarity_search_by_vector_with_relevance_scores(
        vec, k=max(top_k * 4, 40), filter=where_meta
    )
    # 2) fallback روی where_document (contains) فقط اگر خیلی کم بود
    if len(doc_scores) < max(10, top_k):
        doc_scores = vectordb.similarity_search_by_vector_with_relevance_scores(
            vec, k=max(top_k * 6, 60), where_document={"$contains": q_norm[:64]}
        )

    if not doc_scores:
        return {}

    docs, dists = zip(*doc_scores)
    sims = [_to_similarity_from_distance(d) for d in dists]  # تبدیل فاصله→شباهت

    corpus  = [d.page_content for d in docs]
    metas   = [d.metadata for d in docs]
    titles  = [m.get("title","") for m in metas]
    carms   = [m.get("car_model","") for m in metas]
    ids     = [d.id for d in docs]
    urls    = [m.get("url","") for m in metas]
    model_ids = [m.get("model_id","") for m in metas]
    imgs_all  = [ [x.strip() for x in (m.get("images","") or "").split(",") if x.strip()] for m in metas ]
    page_ids  = [m.get("page_id") for m in metas]

    # 3) BM25 روی همان کاندیداها (خیلی ارزان چون n کوچک است)
    query_tokens = parsed.get("tokens") or _tokenize(user_query)
    if _USE_BM25S:
        # bm25s توقع دارد لیست لیست توکن
        bm = BM25S([c.split() for c in corpus])
        bm_scores = bm.get_scores(query_tokens)
    else:
        bm25 = BM25Okapi([c.split() for c in corpus])
        bm_scores = bm25.get_scores(query_tokens)

    # 4) بونس کوچک: اگر نام مدل/عنوان با aliasها همپوشانی دارد
    def _overlap(text: str, toks: List[str]) -> int:
        tl = text.lower()
        return sum(1 for t in toks if t and t.lower() in tl)

    bonuses = []
    for i in range(len(ids)):
        kw_bonus = _overlap(corpus[i], query_tokens) * gamma
        soft = 0.0
        if aliases:
            soft += 1.0 if any(a in (carms[i] or "").lower() for a in aliases) else 0.0
            soft += 0.5 if any(a in (titles[i] or "").lower() for a in aliases) else 0.0
        bonuses.append(kw_bonus + soft)

    packed: List[Tuple] = []
    for i, cid in enumerate(ids):
        fused = alpha * sims[i] + beta * float(bm_scores[i]) + bonuses[i]
        packed.append((
            cid,                 # 0
            titles[i],           # 1
            corpus[i],           # 2
            float(fused),        # 3
            imgs_all[i],         # 4
            model_ids[i],        # 5
            urls[i],             # 6
            "car_spec",          # 7
            page_ids[i],         # 8
        ))
    # مرتب‌سازی نهایی
    packed.sort(key=lambda x: x[3], reverse=True)

    # گروه‌بندی بر اساس مدل‌های تشخیص‌داده‌شده؛ اگر نبود، کلید "ALL"
    bucket_key = "ALL"
    out: Dict[str, List[Tuple]] = {}
    if parsed.get("models_hit"):
        # برای هر مدل مچ‌شده، موارد مرتبط را انتخاب می‌کنیم
        aliases_l = set(aliases)
        for canon in parsed["models_hit"]:
            sel = [t for t in packed if (t[5] and any(a in (t[5] or "").lower() for a in aliases_l)) or
                                      (t[1] and any(a in (t[1] or "").lower() for a in aliases_l))]
            out[canon] = sel[:top_k] if sel else packed[:top_k]
    else:
        out[bucket_key] = packed[:top_k]
    return out

# ───────────────────────────────────────────────────────────────────────────────
# (اختیاری) اگر بخواهید پاسخ نهایی را خارج از Agent بسازید:
# ───────────────────────────────────────────────────────────────────────────────
def gen_answer_prompt(user_query: str, docs: list):
    doc_block = "\n\n".join(
        f"**Chunk ID**: {cid}\n"
        f"**Title**: {title}\n"
        f"{content}\n"
        f"**Source**: {source_name} ({model_id})\n"
        f"**URL**: {url}\n"
        f"**Images**: {', '.join(imgs) if imgs else 'بدون تصویر'}"
        for cid, title, content, _, imgs, model_id, url, source_name in docs
    )
    return f"""
You are an automotive assistant. Respond strictly based on the information provided.

**User question**
{user_query}

**Relevant chunks**
{doc_block}

**Instructions**
* پاسخ را فقط بر اساس همین چانک‌ها بده.
* اگر اطلاعات نبود، بگو نداریم.
* خروجی JSON:
{{"response":"...", "sources":["url1","url2"]}}
"""

def generate_answer(user_query: str, docs: list):
    llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=settings.OPENAI_API_KEY)
    prompt = gen_answer_prompt(user_query, docs)
    from langchain_core.messages import SystemMessage
    resp = llm.invoke([SystemMessage(content=prompt)]).content
    return resp, [], []
