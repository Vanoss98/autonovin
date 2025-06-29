import os, json, re
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from .prompts import ANALYSIS_PROMPT
from ingestion.service import chroma_client
from typing import List, Tuple
import itertools, unicodedata
import json, logging
from typing import Dict, List


EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 1024
EMBED = OpenAIEmbeddings(model=EMBED_MODEL, dimensions=EMBED_DIM)

vectordb = Chroma(
    client=chroma_client,
    collection_name="car_spec",
    embedding_function=EMBED,
)


def analyze_query(user_query: str) -> Dict[str, Dict[str, List[str]]]:
    """
    :returns: {
      "BMW X5": {"aliases": ["BMW X5","بی‌ام‌و X5"], "keywords": ["0-100","گیربکس"]},
      "Audi A3L": {"aliases": ["Audi A3L","آئودی A3L"], "keywords": ["شتاب"]},
      …
    }
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
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
            aliases = list(dict.fromkeys(info.get("aliases", []) + [model]))
            keywords = list(dict.fromkeys(info.get("keywords", [])))
            clean[model] = {"aliases": aliases, "keywords": keywords}
        return clean
    except Exception as e:
        logging.warning("analyze_query failed to parse JSON: %s → %s", e, raw[:200])
        return {}


def _flatten(xs):
    return list(itertools.chain.from_iterable(xs))

def _keyword_overlap(text: str, kws: List[str]) -> int:
    tl = text.lower()
    return sum(1 for kw in kws if kw.lower() in tl)

def hybrid_search(
    analysis: Dict[str, Dict[str, List[str]]],
    *,
    top_k: int = 5,
    alpha: float = 0.45,
    beta: float = 0.40,
    gamma: float = 0.15,
) -> Dict[str, List[Tuple]]:
    """
    :param analysis: output of analyze_query()
    :returns: {
      "BMW X5": [
         (chunk_id, title, content, score, imgs, model_id, url, "car_spec"), …
      ],
      …
    }
    """
    results: Dict[str, List[Tuple]] = {}

    for model, info in analysis.items():
        aliases = info["aliases"]
        kws     = info["keywords"]
        if not kws:
            continue

        # 1) embed only aliases + keywords
        to_embed = " ".join(aliases + kws)
        vec = EMBED.embed_query(to_embed)

        # 2) ANN search *filtered* by metadata.car_model ∈ aliases
        filter = {"car_model": {"$in": aliases}}
        doc_scores = vectordb.similarity_search_by_vector_with_relevance_scores(
            vec, k=top_k * 3, filter=filter
        )
        if not doc_scores:
            results[model] = []
            continue

        docs, vec_scores = zip(*doc_scores)
        corpus = [d.page_content for d in docs]
        metas  = [d.metadata     for d in docs]
        ids    = [d.id           for d in docs]

        # 3) BM25 on the mini-corpus using only keywords
        bm25   = BM25Okapi([c.split() for c in corpus])
        bm_scores = bm25.get_scores(kws)

        # 4) keyword-overlap bonus
        bonus = [_keyword_overlap(c, kws) * gamma for c in corpus]

        # 5) fuse & pack
        packed = []
        for i, cid in enumerate(ids):
            fused = alpha * bm_scores[i] + beta * vec_scores[i] + bonus[i]
            m = metas[i]
            imgs = [x.strip() for x in m.get("images","").split(",") if x.strip()]
            packed.append(
                (
                    cid,
                    m.get("title",""),
                    corpus[i],
                    fused,
                    imgs,
                    m.get("model_id",""),
                    m.get("url",""),
                    "car_spec",
                )
            )

        packed.sort(key=lambda x: x[3], reverse=True)
        results[model] = packed[:top_k]

    return results


def gen_answer_prompt(user_query: str, docs: list):
    """Assemble the full prompt for the LLM."""
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

* Provide responses strictly based on the chunks provided; do not include external knowledge.
* If the user requests a comparison between two models, present a concise comparison using either a brief paragraph or a simple table.
* Write in clear Farsi, maintaining original technical terms (e.g., hp, Nm).
* If the requested information is **not found** in the chunks provided, explicitly state that you don't have the information, and do **not** include any sources.
* Be concise without extra commentary or process explanations.
* Provide your response strictly in JSON format:

{{
  "response": "your response based on chunks provided",
  "sources": ["source_url1", "source_url2", ...]
}}

* Include the sources **only** if your response is based on provided chunks. If no relevant information is available, your JSON should strictly be:

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

    # 3) Parse out the chunk_ids and images used from the final lines
    chunk_ids_used = []
    images_used = []

    # --- Parse citations  ----------------------------------------------------
    chunk_ids_used, urls_used = [], []
    m_chunks = re.search(r"\[(.*?)\]", answer_raw, re.S)
    m_urls = re.search(r"\{(.*?)\}", answer_raw, re.S)
    if m_chunks:
        chunk_ids_used = [c.strip() for c in m_chunks.group(1).split(',') if c.strip()]
    if m_urls:
        urls_used = [u.strip() for u in m_urls.group(1).split(',') if u.strip()]

    # Strip citation lines from displayed answer
    answer_clean = answer_raw.split("\n[")[0].rstrip()

    return answer_clean, chunk_ids_used, urls_used
