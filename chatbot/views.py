# views.py
import json
import logging
import traceback
from collections import defaultdict
from typing import Dict, List
from urllib.parse import urlparse, urlunparse

from django.conf import settings
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from crawler.infrastructure.models import PageImage
from .agent import run_turn

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

BASE_MEDIA_ORIGIN = "https://llm.autonovin.ir"

def _force_origin(u: str) -> str:
    if not u:
        return u
    try:
        p = urlparse(u)
        if p.scheme in ("http", "https") and p.netloc:
            origin = urlparse(BASE_MEDIA_ORIGIN)
            new_parts = (
                origin.scheme or "https",
                origin.netloc,
                p.path if p.path.startswith("/") else f"/{p.path}",
                p.params,
                p.query,
                p.fragment,
            )
            return urlunparse(new_parts)
        path = u if u.startswith("/") else f"/{u}"
        origin = urlparse(BASE_MEDIA_ORIGIN)
        new_parts = (origin.scheme or "https", origin.netloc, path, "", "", "")
        return urlunparse(new_parts)
    except Exception:
        cleaned = (u or "").lstrip("/")
        return f"{BASE_MEDIA_ORIGIN}/{cleaned}"

def _role_of_langchain_msg(msg) -> str:
    t = getattr(msg, "type", "").lower()
    if t == "human": return "user"
    if t == "ai":    return "assistant"
    if t == "tool":  return "tool"
    if t == "system":return "system"
    return t or msg.__class__.__name__.replace("Message", "").lower()

def _content_to_raw_str(raw) -> str:
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw, ensure_ascii=False)
    except Exception:
        return str(raw)

def _annotate_turn_ids_inplace(history_ua: list[dict]) -> None:
    turn_id = 0
    pending_user = None
    for m in history_ua:
        r = m.get("role")
        if r == "user":
            pending_user = m
        elif r == "assistant":
            turn_id += 1
            if pending_user is not None and "turn_id" not in pending_user:
                pending_user["turn_id"] = turn_id
            m["turn_id"] = turn_id
            pending_user = None

def _build_page_images_map(all_page_ids: list[int], request) -> dict[int, list[str]]:
    if not all_page_ids:
        return {}
    qs = PageImage.objects.filter(page_id__in=all_page_ids).select_related("page")
    grouped = defaultdict(list)
    for img in qs:
        if img.file:
            grouped[img.page_id].append(request.build_absolute_uri(img.file.url))
    return {pid: urls for pid, urls in grouped.items()}

# ── NEW: normalization used to match user text to model keys ────────────────
import re as _re
def _norm_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = s.replace("ي","ی").replace("ك","ک")
    s = _re.sub(r"[\u0640\u200c\u200f\u2060]", "", s)
    s = _re.sub(r"\s+", " ", s)
    en2fa = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
    fa2en = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
    # return a combined variant that contains both digit systems for naive containment
    a = s.translate(fa2en)
    b = s.translate(en2fa)
    # prefer ascii-digit variant for substring checks
    return a + " | " + b

# ── NEW: parse full tool payload (we need docs grouped by model) ────────────
def _parse_tool_full(raw: str) -> dict:
    try:
        payload = json.loads(raw or "{}")
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}

# ── NEW: find nearest tool and nearest user before a given assistant idx ────
def _nearest_tool_before_bounded(full_linear: list[dict], *, assistant_idx: int) -> str:
    j = assistant_idx - 1
    while j >= 0:
        role = full_linear[j].get("role")
        if role == "assistant":
            return ""  # bounded at previous assistant
        if role == "tool":
            return full_linear[j].get("content_raw", "")
        j -= 1
    return ""

def _nearest_user_before(full_linear: list[dict], *, assistant_idx: int) -> str:
    j = assistant_idx - 1
    while j >= 0:
        if full_linear[j].get("role") == "user":
            return full_linear[j].get("content_raw", "")
        j -= 1
    return ""

# ── NEW: choose primary model from tool docs + user/assistant texts ─────────
def _pick_primary_model_from_tool(payload: dict, user_text: str, assistant_text: str) -> str | None:
    docs = payload.get("docs")
    if not isinstance(docs, dict) or not docs:
        return None
    user_n = _norm_text(user_text)
    asst_n = _norm_text(assistant_text)
    models = list(docs.keys())
    if not models:
        return None

    # 1) direct containment: if model key appears in user or assistant text
    candidates = []
    for m in models:
        m_n_a = _norm_text(m)
        score = 0
        if m_n_a and m_n_a.split(" | ")[0] in user_n: score += 3
        if m_n_a and m_n_a.split(" | ")[1] in user_n: score += 3
        if m_n_a and m_n_a.split(" | ")[0] in asst_n: score += 2
        if m_n_a and m_n_a.split(" | ")[1] in asst_n: score += 2
        if score > 0:
            candidates.append((score, m))

    if candidates:
        candidates.sort(reverse=True)  # highest score first
        return candidates[0][1]

    # 2) fallback: the model with the largest number of tuples
    return max(models, key=lambda m: len(docs.get(m) or []), default=None)

# ── NEW: build urls/page_ids for ONLY one selected model; allow intersect with assistant urls ──
def _urls_and_pids_for_model(payload: dict, model: str, prefer_urls: List[str] | None = None) -> tuple[List[str], List[int]]:
    docs = payload.get("docs") or {}
    lst = docs.get(model) or []
    # extract ordered urls/pids from tuples: url at 6, page_id at 8
    urls = []
    pids = []
    seen_u, seen_p = set(), set()
    for t in lst:
        if isinstance(t, (list, tuple)):
            u = t[6] if len(t) >= 7 else None
            pid = t[8] if len(t) >= 9 else None
            if isinstance(u, str) and u and u not in seen_u:
                seen_u.add(u); urls.append(u)
            if isinstance(pid, int) and pid not in seen_p:
                seen_p.add(pid); pids.append(pid)

    if prefer_urls:
        # Intersection while preserving assistant order first, then keep remaining model URLs
        ordered_pref = [u for u in prefer_urls if u in seen_u]
        remaining = [u for u in urls if u not in ordered_pref]
        final_urls = ordered_pref + remaining
        # derive pids from final_urls order
        url_to_pid = {}
        for t in lst:
            if isinstance(t, (list, tuple)) and len(t) >= 9:
                uu, pp = t[6], t[8]
                if isinstance(uu, str) and isinstance(pp, int) and uu not in url_to_pid:
                    url_to_pid[uu] = pp
        final_pids = []
        seenp = set()
        for u in final_urls:
            pid = url_to_pid.get(u)
            if isinstance(pid, int) and pid not in seenp:
                seenp.add(pid); final_pids.append(pid)
        return final_urls, final_pids

    return urls, pids

@method_decorator(csrf_exempt, name="dispatch")
class ChatAPIView(APIView):
    def post(self, request):
        user_text = request.data.get("message")
        if not user_text:
            return Response({"error": "Field 'message' is required."},
                            status=status.HTTP_400_BAD_REQUEST)

        thread_id_in = request.data.get("thread_id")
        try:
            answer, urls_current, thread_id_out, _ = run_turn(user_text, thread_id=thread_id_in)
            from .agent import get_thread_messages
            msgs = get_thread_messages(thread_id_out) or []
        except Exception as exc:
            logger.exception("Chatbot crashed")
            if settings.DEBUG:
                return Response(
                    {"error": str(exc), "type": exc.__class__.__name__, "traceback": traceback.format_exc()},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response({"error": "Internal Server Error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # 1) Linearize messages; skip empty assistant stubs
        linear = []
        for m in msgs:
            role = _role_of_langchain_msg(m)
            raw = _content_to_raw_str(m.content)
            if role == "assistant" and not (raw or "").strip():
                continue
            linear.append({"role": role, "content_raw": raw})

        # 2) Build history with model-aware selection for images/sources
        assistant_indices = [i for i, _ in enumerate(linear) if linear[i]["role"] == "assistant"]
        per_assistant_page_ids: Dict[int, List[int]] = {}
        per_assistant_sources: Dict[int, List[str]] = {}
        all_page_ids: List[int] = []

        for ai in assistant_indices:
            # assistant JSON (if any)
            content_disp = linear[ai]["content_raw"]
            urls_from_assistant: List[str] = []
            try:
                payload_asst = json.loads(content_disp)
                if isinstance(payload_asst, dict) and "response" in payload_asst:
                    content_disp = payload_asst.get("response", content_disp)
                    urls_from_assistant = payload_asst.get("urls", []) or []
            except Exception:
                pass
            linear[ai]["content_disp"] = content_disp

            # bounded tool payload (needed to map model->(urls,pids))
            tool_raw = _nearest_tool_before_bounded(linear, assistant_idx=ai)
            tool_payload = _parse_tool_full(tool_raw)

            # nearest user text for this turn (to help pick the model)
            user_q = _nearest_user_before(linear, assistant_idx=ai)

            # choose primary model from tool docs + user/assistant text
            primary = _pick_primary_model_from_tool(tool_payload, user_q, content_disp)

            # build urls/pids for that model; bias to assistant urls if provided
            if primary:
                urls_sel, pids_sel = _urls_and_pids_for_model(tool_payload, primary, prefer_urls=urls_from_assistant)
            else:
                # fallback: nothing model-aware available — keep empty (no wrong images)
                urls_sel, pids_sel = [], []

            per_assistant_sources[ai] = urls_sel
            per_assistant_page_ids[ai] = pids_sel
            all_page_ids.extend(pids_sel)

        # 3) Fetch images for selected page_ids only
        all_page_ids = list(dict.fromkeys(all_page_ids))
        page_images_map = _build_page_images_map(all_page_ids, request)

        # 4) Build UA history
        history_ua = []
        for i, rec in enumerate(linear):
            role = rec["role"]
            if role in ("tool", "system"):
                continue
            if role == "user":
                history_ua.append({"role": "user", "content": rec["content_raw"]})
            elif role == "assistant":
                pids = per_assistant_page_ids.get(i, []) or []
                imgs: List[str] = []
                for pid in pids:
                    imgs.extend(page_images_map.get(pid, []))
                imgs = list(dict.fromkeys(_force_origin(x) for x in imgs))
                history_ua.append({
                    "role": "assistant",
                    "content": rec.get("content_disp", rec["content_raw"]),
                    "images": imgs,
                    "sources": per_assistant_sources.get(i, []) or [],
                })

        _annotate_turn_ids_inplace(history_ua)

        return Response({
            "thread_id": thread_id_out,
            "answer": answer,
            "history": history_ua,
        }, status=status.HTTP_200_OK)


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/chat/<thread_id>/  →  return full history (images/sources per turn)
# ──────────────────────────────────────────────────────────────────────────────

@method_decorator(csrf_exempt, name="dispatch")
class ChatHistoryAPIView(APIView):
    def get(self, request, thread_id: str):
        from .agent import get_thread_messages
        msgs = get_thread_messages(thread_id) or []
        if not msgs:
            return Response({"thread_id": thread_id, "history": []}, status=status.HTTP_200_OK)

        linear = []
        for m in msgs:
            role = _role_of_langchain_msg(m)
            raw = _content_to_raw_str(m.content)
            if role == "assistant" and not (raw or "").strip():
                continue
            linear.append({"role": role, "content_raw": raw})

        assistant_indices = [i for i, _ in enumerate(linear) if linear[i]["role"] == "assistant"]
        per_assistant_page_ids: Dict[int, List[int]] = {}
        per_assistant_sources: Dict[int, List[str]] = {}
        all_page_ids: List[int] = []

        for ai in assistant_indices:
            content_disp = linear[ai]["content_raw"]
            urls_from_assistant: List[str] = []
            try:
                payload_asst = json.loads(content_disp)
                if isinstance(payload_asst, dict) and "response" in payload_asst:
                    content_disp = payload_asst.get("response", content_disp)
                    urls_from_assistant = payload_asst.get("urls", []) or []
            except Exception:
                pass
            linear[ai]["content_disp"] = content_disp

            tool_raw = _nearest_tool_before_bounded(linear, assistant_idx=ai)
            tool_payload = _parse_tool_full(tool_raw)

            user_q = _nearest_user_before(linear, assistant_idx=ai)
            primary = _pick_primary_model_from_tool(tool_payload, user_q, content_disp)

            if primary:
                urls_sel, pids_sel = _urls_and_pids_for_model(tool_payload, primary, prefer_urls=urls_from_assistant)
            else:
                urls_sel, pids_sel = [], []

            per_assistant_sources[ai] = urls_sel
            per_assistant_page_ids[ai] = pids_sel
            all_page_ids.extend(pids_sel)

        all_page_ids = list(dict.fromkeys(all_page_ids))
        page_images_map = _build_page_images_map(all_page_ids, request)

        history_ua = []
        for i, rec in enumerate(linear):
            role = rec["role"]
            if role in ("tool", "system"):
                continue
            if role == "user":
                history_ua.append({"role": "user", "content": rec["content_raw"]})
            elif role == "assistant":
                pids = per_assistant_page_ids.get(i, []) or []
                imgs: List[str] = []
                for pid in pids:
                    imgs.extend(page_images_map.get(pid, []))
                imgs = list(dict.fromkeys(_force_origin(x) for x in imgs))
                history_ua.append({
                    "role": "assistant",
                    "content": rec.get("content_disp", rec["content_raw"]),
                    "images": imgs,
                    "sources": per_assistant_sources.get(i, []) or [],
                })

        _annotate_turn_ids_inplace(history_ua)
        return Response({
            "thread_id": thread_id,
            "history": history_ua,
        }, status=status.HTTP_200_OK)

# @method_decorator(csrf_exempt, name="dispatch")
# class AnalyzeQueryAPIView(APIView):
#     """
#     Test endpoint for query analysis, retrieval, AND final LLM response.
#     Returns analysis results, retrieved documents, and generated answer.
#     """
#
#     def post(self, request):
#         query = request.data.get("query")
#         if not query:
#             return Response(
#                 {"error": "Field 'query' is required."},
#                 status=status.HTTP_400_BAD_REQUEST
#             )
#
#         # Optional parameters for retrieval
#         top_k = request.data.get("top_k", 5)
#         alpha = request.data.get("alpha", 0.45)
#         beta = request.data.get("beta", 0.40)
#         gamma = request.data.get("gamma", 0.15)
#         show_full_content = request.data.get("show_full_content", False)
#         include_llm_response = request.data.get("include_llm_response", True)
#
#         # Timing tracking
#         import time
#         timings = {}
#         start_total = time.time()
#
#         try:
#             from .utils import analyze_query, hybrid_search, generate_answer
#
#             # Step 1: Query Analysis
#             start_step = time.time()
#             analysis_result = analyze_query(query)
#             timings["analysis"] = round(time.time() - start_step, 3)
#
#             # Format analysis for display
#             formatted_analysis = {}
#             for model, info in analysis_result.items():
#                 formatted_analysis[model] = {
#                     "aliases": info.get("aliases", []),
#                     "keywords": info.get("keywords", [])
#                 }
#
#             # Step 2: Hybrid Retrieval
#             start_step = time.time()
#             retrieval_result = hybrid_search(
#                 analysis_result,
#                 top_k=top_k,
#                 alpha=alpha,
#                 beta=beta,
#                 gamma=gamma
#             )
#             timings["retrieval"] = round(time.time() - start_step, 3)
#
#             # Format retrieval results and prepare for LLM (like ChatAPIView)
#             formatted_retrieval = {}
#             all_urls = set()
#             all_page_ids = set()
#             total_docs = 0
#
#             # Collect docs in the format ChatAPIView uses (8-element tuples without page_id)
#             all_docs_for_llm = []
#
#             for model, docs in retrieval_result.items():
#                 formatted_docs = []
#                 for doc in docs:
#                     # Unpack document tuple (9 elements with page_id)
#                     doc_data = {
#                         "chunk_id": doc[0],
#                         "title": doc[1],
#                         "content": doc[2] if show_full_content else (
#                             doc[2][:300] + "..." if len(doc[2]) > 300 else doc[2]),
#                         "score": round(doc[3], 4) if isinstance(doc[3], (int, float)) else doc[3],
#                         "images": doc[4] if len(doc) > 4 else [],
#                         "model_id": doc[5] if len(doc) > 5 else None,
#                         "url": doc[6] if len(doc) > 6 else None,
#                         "source": doc[7] if len(doc) > 7 else None,
#                         "page_id": doc[8] if len(doc) > 8 else None
#                     }
#
#                     formatted_docs.append(doc_data)
#
#                     # Convert to 8-element tuple for LLM (exclude page_id like ChatAPIView does)
#                     doc_for_llm = (
#                         doc[0],  # chunk_id
#                         doc[1],  # title
#                         doc[2],  # content
#                         doc[3],  # score
#                         doc[4] if len(doc) > 4 else [],  # images
#                         doc[5] if len(doc) > 5 else None,  # model_id
#                         doc[6] if len(doc) > 6 else None,  # url
#                         doc[7] if len(doc) > 7 else None  # source_name
#                         # page_id (doc[8]) is intentionally excluded
#                     )
#                     all_docs_for_llm.append(doc_for_llm)
#
#                     # Collect unique URLs and page_ids
#                     if doc_data["url"]:
#                         all_urls.add(doc_data["url"])
#                     if doc_data["page_id"]:
#                         all_page_ids.add(doc_data["page_id"])
#
#                     total_docs += 1
#
#                 formatted_retrieval[model] = formatted_docs
#
#             # Step 3: Generate LLM Response (if requested)
#             llm_response_data = None
#             if include_llm_response and all_docs_for_llm:
#                 start_step = time.time()
#                 try:
#                     answer_clean, chunk_ids_used, urls_used = generate_answer(
#                         query,
#                         all_docs_for_llm  # Now passing 8-element tuples like ChatAPIView
#                     )
#                     timings["llm_generation"] = round(time.time() - start_step, 3)
#
#                     llm_response_data = {
#                         "answer": answer_clean,
#                         "chunk_ids_used": chunk_ids_used,
#                         "urls_used": urls_used,
#                         "generation_time_seconds": timings["llm_generation"]
#                     }
#                 except Exception as llm_exc:
#                     logger.exception("LLM generation failed")
#                     timings["llm_generation"] = round(time.time() - start_step, 3)
#                     llm_response_data = {
#                         "error": str(llm_exc),
#                         "error_type": llm_exc.__class__.__name__,
#                         "generation_time_seconds": timings["llm_generation"]
#                     }
#                     if settings.DEBUG:
#                         llm_response_data["traceback"] = traceback.format_exc()
#
#             timings["total"] = round(time.time() - start_total, 3)
#
#             # Calculate statistics
#             stats = {
#                 "models_analyzed": len(formatted_analysis),
#                 "total_documents_retrieved": total_docs,
#                 "documents_per_model": {model: len(docs) for model, docs in formatted_retrieval.items()},
#                 "unique_urls_count": len(all_urls),
#                 "unique_page_ids_count": len(all_page_ids),
#                 "avg_docs_per_model": round(total_docs / len(formatted_analysis), 2) if formatted_analysis else 0
#             }
#
#             # Get score distribution for each model
#             score_distribution = {}
#             for model, docs in formatted_retrieval.items():
#                 if docs:
#                     scores = [d["score"] for d in docs if isinstance(d["score"], (int, float))]
#                     if scores:
#                         score_distribution[model] = {
#                             "max": round(max(scores), 4),
#                             "min": round(min(scores), 4),
#                             "avg": round(sum(scores) / len(scores), 4),
#                             "count": len(scores)
#                         }
#
#             response_data = {
#                 "query": query,
#                 "timestamp": datetime.now().isoformat(),
#
#                 # Performance Metrics
#                 "performance": {
#                     "timings_seconds": timings,
#                     "bottleneck": max(timings.items(), key=lambda x: x[1])[0] if timings else None,
#                     "bottleneck_time": max(timings.values()) if timings else None,
#                     "bottleneck_percentage": round((max(timings.values()) / timings["total"] * 100), 1) if timings.get(
#                         "total") else None
#                 },
#
#                 # Analysis Results
#                 "analysis": {
#                     "models": formatted_analysis,
#                     "model_count": len(formatted_analysis),
#                     "total_keywords": sum(len(info["keywords"]) for info in formatted_analysis.values()),
#                     "total_aliases": sum(len(info["aliases"]) for info in formatted_analysis.values())
#                 },
#
#                 # Retrieval Results
#                 "retrieval": {
#                     "documents": formatted_retrieval,
#                     "parameters": {
#                         "top_k": top_k,
#                         "alpha_bm25": alpha,
#                         "beta_vector": beta,
#                         "gamma_keyword": gamma
#                     },
#                     "score_distribution": score_distribution
#                 },
#
#                 # LLM Response (if generated)
#                 "llm_response": llm_response_data,
#
#                 # Summary Statistics
#                 "statistics": stats,
#
#                 # Unique Resources
#                 "resources": {
#                     "urls": sorted(list(all_urls)),
#                     "page_ids": sorted(list(all_page_ids))
#                 }
#             }
#
#             # Add debug info if requested
#             if request.data.get("debug"):
#                 response_data["debug"] = {
#                     "retrieval_method": "hybrid_search",
#                     "vector_model": "text-embedding-3-large",
#                     "chroma_collection": "car_spec",
#                     "total_docs_sent_to_llm": len(all_docs_for_llm),
#                     "doc_tuple_format": "8-element (chunk_id, title, content, score, images, model_id, url, source_name)"
#                 }
#
#             return Response(response_data, status=status.HTTP_200_OK)
#
#         except Exception as exc:
#             logger.exception("Analysis and retrieval failed")
#             error_response = {
#                 "error": "Failed to analyze and retrieve",
#                 "query": query,
#                 "timestamp": datetime.now().isoformat(),
#                 "timings_before_failure": timings
#             }
#
#             if settings.DEBUG:
#                 error_response.update({
#                     "exception": str(exc),
#                     "type": exc.__class__.__name__,
#                     "traceback": traceback.format_exc()
#                 })
#
#             return Response(
#                 error_response,
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )