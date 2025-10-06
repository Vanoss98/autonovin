# utils.py
import os, re, json, itertools, logging, unicodedata
from typing import Dict, List, Tuple, Any
from functools import lru_cache

from django.conf import settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# BM25: اگر bm25s نصب باشد سریع‌تر است؛ در غیر اینصورت rank_bm25
_USE_BM25S = False
try:
    from bm25s import BM25 as BM25S  # pip install bm25s
    _USE_BM25S = True
except Exception:
    from rank_bm25 import BM25Okapi
    BM25S = None

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import SystemMessage

from .prompts import ANALYSIS_PROMPT
from ingestion.service import chroma_client

set_llm_cache(InMemoryCache())

# ───────── Embedding (بدون تغییر مدل) ─────────
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM   = 1024
EMBED       = OpenAIEmbeddings(model=EMBED_MODEL, dimensions=EMBED_DIM)

@lru_cache(maxsize=4096)
def _embed_cached(q: str) -> List[float]:
    return EMBED.embed_query(q)

# ───────── VectorDB ─────────
vectordb = Chroma(
    client=chroma_client,
    collection_name="car_spec",
    embedding_function=EMBED,
)

# ───────── نرمال‌سازی ─────────
_TATWEEL_RE = re.compile(r"[\u0640\u200c\u200f\u2060]")
_MULTI_WS_RE = re.compile(r"\s+")
_FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_EN_DIGITS = "0123456789"
_EN2FA = str.maketrans(_EN_DIGITS, _FA_DIGITS)
_FA2EN = str.maketrans(_FA_DIGITS, _EN_DIGITS)

def _normalize(s: str) -> str:
    if not s: return ""
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
    toks = re.split(r"[^\w\u0600-\u06FF\-\+\.%/]+", s)
    return [t for t in toks if t]

# ───────── تحلیل پرسش (LLM) — سریع و قابل کش ─────────
# همان مدل: gpt-4.1. JSON Mode برای خروجی قابل‌اعتماد.
_LLM_ANALYZE = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
    timeout=15,          # ↓ تاخیر
    max_retries=2,       # ↓ retry
    api_key=settings.OPENAI_API_KEY,
).bind(response_format={"type": "json_object"})  # JSON Mode  :contentReference[oaicite:4]{index=4}

def _safe_json_loads(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        # تلاش برای بیرون کشیدن بزرگ‌ترین بلاک JSON
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
    return {}

@lru_cache(maxsize=2048)
def _analyze_cached(norm_query: str) -> Dict[str, Dict[str, List[str]]]:
    prompt = ANALYSIS_PROMPT.format(user_query=norm_query)
    raw = _LLM_ANALYZE.invoke([SystemMessage(content=prompt)]).content.strip()
    data = _safe_json_loads(raw)
    clean = {}
    if isinstance(data, dict):
        for model, info in data.items():
            aliases = list(dict.fromkeys((info.get("aliases") or []) + [model]))
            keywords = list(dict.fromkeys(info.get("keywords") or []))
            clean[model] = {"aliases": aliases, "keywords": keywords}
    return clean

def analyze_query(user_query: str) -> Dict[str, Dict[str, List[str]]]:
    """همان API قبلی؛ فقط کش و JSON-mode اضافه شده."""
    norm = _normalize(user_query)
    try:
        data = _analyze_cached(norm)
        return data or {}
    except Exception as e:
        logging.warning("analyze_query failed: %s", e)
        return {}

# ───────── ابزارهای کمکی Hybrid ─────────
def _norm(s: str) -> set[str]:
    if not s: return {""}
    s = _normalize(s)
    return { s.translate(_FA2EN), s.translate(_EN2FA) }

def _soft_contains(hay: str, aliases_norm: set[str]) -> bool:
    h = next(iter(_norm(hay)))
    return any(a in h for a in aliases_norm)

def _to_similarity_from_distance(d: float) -> float:
    # در Chroma «distance» هرچه کمتر، مشابهت بیشتر است.  :contentReference[oaicite:5]{index=5}
    return 1.0 / (1.0 + max(float(d), 0.0))

# ───────── Hybrid Search (همان متد، بهینه‌شده) ─────────
def hybrid_search(
    analysis: Dict[str, Dict[str, List[str]]],
    *,
    top_k: int = 5,
    alpha: float = 0.45,
    beta: float = 0.40,
    gamma: float = 0.15,
) -> Dict[str, List[Tuple]]:
    results: Dict[str, List[Tuple]] = {}

    for model, info in (analysis or {}).items():
        raw_aliases = list(dict.fromkeys((info.get("aliases") or []) + [model]))
        kws = info.get("keywords", [])

        # نرمال‌سازی aliasها (هر دو فرم عددی)
        aliases_norm = set()
        for a in raw_aliases:
            aliases_norm |= _norm(a)
        if not aliases_norm:
            results[model] = []
            continue

        # یک بار Embed (کش‌پذیر) — رشتهٔ کوتاه‌تر برای هزینهٔ کمتر
        to_embed = " ".join(sorted(list(aliases_norm))[:6] + kws[:6])  # cap
        vec = _embed_cached(to_embed)

        # 1) فیلتر متادیتا (سریع) — مستند Chroma برای where/where_document  :contentReference[oaicite:6]{index=6}
        where_meta = {"car_model": {"$in": list(aliases_norm)}}
        doc_scores = vectordb.similarity_search_by_vector_with_relevance_scores(
            vec, k=max(top_k * 3, 30), filter=where_meta
        )

        # 2) fallback: یک یا دو تلاش روی طولانی‌ترین alias بجای حلقهٔ طولانی
        if not doc_scores:
            sorted_aliases = sorted(aliases_norm, key=len, reverse=True)[:2]
            for a in sorted_aliases:
                doc_scores = vectordb.similarity_search_by_vector_with_relevance_scores(
                    vec, k=max(top_k * 5, 40), where_document={"$contains": a}
                )
                if doc_scores: break

        # 3) آخرین fallback: ANN بدون فیلتر
        if not doc_scores:
            doc_scores = vectordb.similarity_search_by_vector_with_relevance_scores(
                vec, k=max(top_k * 5, 40)
            )
        if not doc_scores:
            results[model] = []
            continue

        docs, dists = zip(*doc_scores)
        sims = [_to_similarity_from_distance(d) for d in dists]

        corpus = [d.page_content for d in docs]
        metas  = [d.metadata     for d in docs]
        ids    = [d.id           for d in docs]
        titles = [m.get("title","") for m in metas]
        carms  = [m.get("car_model","") for m in metas]

        # BM25 روی همین کاندیداها (خیلی ارزان) — اگر bm25s در دسترس باشد، خودکار
        query_tokens = kws if kws else list(aliases_norm)
        if _USE_BM25S and len(corpus) > 0:
            bm = BM25S([c.split() for c in corpus])        # :contentReference[oaicite:7]{index=7}
            bm_scores = bm.get_scores(query_tokens)
        else:
            bm25 = BM25Okapi([c.split() for c in corpus])  # :contentReference[oaicite:8]{index=8}
            bm_scores = bm25.get_scores(query_tokens)

        def _overlap(text: str, toks: List[str]) -> int:
            tl = text.lower()
            return sum(1 for t in toks if t.lower() in tl)

        bonuses = []
        for i in range(len(ids)):
            kw_bonus = _overlap(corpus[i], query_tokens) * gamma
            soft = 1.0 if (_soft_contains(carms[i], aliases_norm) or _soft_contains(titles[i], aliases_norm)) else 0.0
            bonuses.append(kw_bonus + soft)

        packed: List[Tuple] = []
        for i, d in enumerate(docs):
            m = metas[i]
            imgs = [x.strip() for x in (m.get("images","") or "").split(",") if x.strip()]
            page_id = m.get("page_id")
            fused = alpha * sims[i] + beta * float(bm_scores[i]) + bonuses[i]
            packed.append((
                d.id,                     # 0
                m.get("title",""),        # 1
                d.page_content,           # 2
                float(fused),             # 3
                imgs,                     # 4
                m.get("model_id",""),     # 5
                m.get("url",""),          # 6
                "car_spec",               # 7
                page_id,                  # 8
            ))

        packed.sort(key=lambda x: x[3], reverse=True)
        results[model] = packed[:top_k]

    return results

# ───────── (سازگاری) تولید پاسخ اگر بیرون از ایجنت خواستید ─────────
def gen_answer_prompt(user_query: str, docs: list):
    doc_block = "\n\n".join(
        f"**Chunk ID**: {cid}\n"
        f"**Title**: {title}\n"
        f"{content}\n"
        f"**Source**: {source_name} ({model_id})\n"
        f"**URL**: {url}\n"
        f"**Images**: {', '.join(imgs) if imgs else 'بدون تصویر'}"
        for cid, title, content, _, imgs, model_id, url, source_name, *_ in docs
    )
    return f"""
You are an automotive assistant. Respond strictly based on the information provided.

**User question**
{user_query}

**Relevant chunks**
{doc_block}

**Instructions**
* Provide responses strictly based on the chunks provided; do not include external knowledge.
* Write in clear Farsi.
* JSON output: {{"response":"...", "sources":["url1","url2"]}}
"""

def generate_answer(user_query: str, docs: list):
    llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=settings.OPENAI_API_KEY)
    sys_message = SystemMessage(content=gen_answer_prompt(user_query, docs))
    response = llm.invoke([sys_message]).content
    return response, [], []
