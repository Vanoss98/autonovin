# utils.py
import itertools
import json
import logging
import re
from typing import Dict, List, Tuple

from django.conf import settings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rank_bm25 import BM25Okapi

from ingestion.service import chroma_client
from .prompts import ANALYSIS_PROMPT

# ──────────────────────────────────────────────────────────────────────────────
# Embedding store
# ──────────────────────────────────────────────────────────────────────────────

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 1024
EMBED = OpenAIEmbeddings(model=EMBED_MODEL, dimensions=EMBED_DIM)

vectordb = Chroma(
    client=chroma_client,
    collection_name="car_spec",
    embedding_function=EMBED,
)

# ──────────────────────────────────────────────────────────────────────────────
# Normalization utils (reused by agent tool)
# ──────────────────────────────────────────────────────────────────────────────

def _flatten(xs):
    return list(itertools.chain.from_iterable(xs))

def _keyword_overlap(text: str, kws: List[str]) -> int:
    tl = text.lower()
    return sum(1 for kw in kws if kw.lower() in tl)

def _norm(s: str) -> set[str]:
    """Return two normalized variants (fa/en digits), collapse ZW characters."""
    if not s:
        return {""}
    s = s.strip().lower()
    s = s.replace("ي", "ی").replace("ك", "ک")  # arabic->persian
    s = re.sub(r"[\u0640\u200c\u200f\u2060]", "", s)  # tatweel + ZW chars
    s = re.sub(r"\s+", " ", s)
    en2fa = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
    fa2en = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
    return {s.translate(fa2en), s.translate(en2fa)}

def _soft_contains(hay: str, aliases_norm: set[str]) -> bool:
    """Substring check on a normalized haystack vs alias variants."""
    h = next(iter(_norm(hay)))
    return any(a in h for a in aliases_norm if a)


# ──────────────────────────────────────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────────────────────────────────────

def analyze_query(user_query: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns mapping:
      {
        "BMW X5": {"aliases": ["BMW X5","بی‌ام‌و X5"], "keywords": ["0-100","گیربکس"]},
        "Audi A3L": {"aliases": ["Audi A3L","آئودی A3L"], "keywords": ["شتاب"]},
        …
      }
    """
    llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.5,
        timeout=45,
        max_retries=3,
        api_key=settings.OPENAI_API_KEY,
    )
    prompt = ANALYSIS_PROMPT.format(user_query=user_query)
    raw = llm.invoke([SystemMessage(content=prompt)]).content.strip()

    try:
        data = json.loads(raw)
        clean = {}
        for model, info in data.items():
            aliases = list(dict.fromkeys((info.get("aliases") or []) + [model]))
            keywords = list(dict.fromkeys(info.get("keywords", [])))
            clean[model] = {"aliases": aliases, "keywords": keywords}
        return clean
    except Exception as e:
        logging.warning("analyze_query failed to parse JSON: %s → %s", e, raw[:200])
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid retrieval (now includes car_model(meta) at index 9)
# ──────────────────────────────────────────────────────────────────────────────

def hybrid_search(
    analysis: Dict[str, Dict[str, List[str]]],
    *,
    top_k: int = 5,
    alpha: float = 0.45,
    beta: float = 0.40,
    gamma: float = 0.15,
) -> Dict[str, List[Tuple]]:
    """
    Returns per-model list of tuples:
      (
        0 id,
        1 title,
        2 content,
        3 fused_score,
        4 images[],            # from metadata "images" (comma-separated originally)
        5 model_id,            # from metadata
        6 url,                 # from metadata
        7 source_name,         # constant "car_spec"
        8 page_id,             # from metadata
        9 car_model_meta       # NEW: from metadata["car_model"]
      )
    """
    results: Dict[str, List[Tuple]] = {}

    for model, info in (analysis or {}).items():
        raw_aliases = list(dict.fromkeys((info.get("aliases") or []) + [model]))
        kws = info.get("keywords", [])  # allow empty

        # normalize all aliases (keep both en/fa-digit forms)
        aliases_norm = set()
        for a in raw_aliases:
            aliases_norm |= _norm(a)
        if not aliases_norm:
            results[model] = []
            continue

        # embed aliases+kws even if kws is empty
        to_embed = " ".join(list(aliases_norm) + kws)
        vec = EMBED.embed_query(to_embed)

        # 1) strict metadata filter by car_model
        where_meta = {"car_model": {"$in": list(aliases_norm)}}
        doc_scores = vectordb.similarity_search_by_vector_with_relevance_scores(
            vec, k=top_k * 3, filter=where_meta
        )

        # 2) fallback: filter by document text substring (longest alias first)
        if not doc_scores:
            for a in sorted(aliases_norm, key=len, reverse=True):
                doc_scores = vectordb.similarity_search_by_vector_with_relevance_scores(
                    vec, k=top_k * 6, where_document={"$contains": a}
                )
                if doc_scores:
                    break

        # 3) last fallback: unfiltered ANN
        if not doc_scores:
            doc_scores = vectordb.similarity_search_by_vector_with_relevance_scores(
                vec, k=top_k * 6
            )
        if not doc_scores:
            results[model] = []
            continue

        docs, vec_scores = zip(*doc_scores)
        corpus = [d.page_content for d in docs]
        metas  = [d.metadata     for d in docs]
        ids    = [d.id           for d in docs]
        titles = [m.get("title","") for m in metas]
        carms  = [m.get("car_model","") for m in metas]

        # BM25 over corpus; if no kws, use alias tokens instead
        query_tokens = kws if kws else list(aliases_norm)
        bm25 = BM25Okapi([c.split() for c in corpus])
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
        for i, cid in enumerate(ids):
            fused = alpha * bm_scores[i] + beta * vec_scores[i] + bonuses[i]
            m = metas[i]
            imgs = [x.strip() for x in (m.get("images","") or "").split(",") if x.strip()]
            page_id = m.get("page_id")
            packed.append(
                (
                    cid,                 # 0
                    titles[i],           # 1
                    corpus[i],           # 2
                    fused,               # 3
                    imgs,                # 4
                    m.get("model_id",""),# 5
                    m.get("url",""),     # 6
                    "car_spec",          # 7
                    page_id,             # 8
                    m.get("car_model","")# 9  <-- NEW: used by model guard
                )
            )

        packed.sort(key=lambda x: x[3], reverse=True)
        results[model] = packed[:top_k]

    return results


# ──────────────────────────────────────────────────────────────────────────────
# (Optional) Generator you keep — unchanged
# ──────────────────────────────────────────────────────────────────────────────

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
* If the user requests a comparison between two models, present a concise comparison using either a brief paragraph or a simple table.
* Write in clear Farsi, maintaining original technical terms (e.g., hp, Nm).
* If the requested information is **not found** in the chunks provided, explicitly state that you don't have the information.
* Be concise without extra commentary or process explanations.
* Provide your response strictly in JSON format:

{{
  "response": "your response based on chunks provided",
  "sources": ["source_url1", "source_url2", ...]
}}

If no relevant information is available:

{{
  "response": "متأسفانه اطلاعات درخواستی در مستندات ارائه‌شده موجود نیست."
}}
"""

def generate_answer(user_query: str, docs: list):
    prompt = gen_answer_prompt(user_query, docs)
    llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=settings.OPENAI_API_KEY)
    sys_message = SystemMessage(content=prompt)
    response = llm.invoke([sys_message])
    answer_raw = response.content

    # Optional parser (not used by the agent path)
    try:
        parsed = json.loads(answer_raw)
        return parsed.get("response", answer_raw), [], parsed.get("sources", [])
    except Exception:
        return answer_raw, [], []
