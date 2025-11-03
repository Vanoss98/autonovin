# views.py
from datetime import datetime

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .agent import run_turn
import logging, traceback, json
from django.conf import settings
from collections import defaultdict
from crawler.infrastructure.models import PageImage
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

from .prompts import ANALYSIS_PROMPT

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _flatten_images(images_by_page: dict[int, list[str]]) -> list[str]:
    out = []
    for urls in images_by_page.values():
        for u in urls:
            if u:
                out.append(u)
    # de-dupe, preserve order
    return list(dict.fromkeys(out))

def _parse_tool_payload(raw: str) -> tuple[list[int], list[str]]:
    """
    Parse a tool's JSON string and return (page_ids, urls).
    Accepts both {"page_ids":[...], "urls":[...]} and legacy {"docs":{...}}.
    """
    page_ids, urls = [], []
    try:
        payload = json.loads(raw or "{}")
        if not isinstance(payload, dict):
            return [], []
        # urls
        if isinstance(payload.get("urls"), list):
            urls = [u for u in payload["urls"] if isinstance(u, str)]
        # page_ids
        if isinstance(payload.get("page_ids"), list):
            page_ids = [pid for pid in payload["page_ids"] if isinstance(pid, int)]
        elif "docs" in payload and isinstance(payload["docs"], dict):
            inferred = []
            for _, lst in payload["docs"].items():
                for tup in (lst or []):
                    if isinstance(tup, (list, tuple)) and len(tup) >= 9 and tup[8] is not None:
                        inferred.append(tup[8])
            page_ids = [pid for pid in dict.fromkeys(inferred)]
    except Exception:
        pass
    return page_ids, urls

def _nearest_tool_before_bounded(full_linear: list[dict], *, assistant_idx: int) -> tuple[list[int], list[str]]:
    """
    Walk backward from assistant_idx-1; stop at previous assistant.
    Returns (page_ids, urls) from that tool, or ([], []).
    'full_linear' is a list of dicts with keys: role, content_raw.
    """
    j = assistant_idx - 1
    while j >= 0:
        role = full_linear[j].get("role")
        if role == "assistant":
            return [], []
        if role == "tool":
            return _parse_tool_payload(full_linear[j].get("content_raw"))
        j -= 1
    return [], []

def _build_page_images_map(all_page_ids: list[int], request) -> dict[int, list[str]]:
    if not all_page_ids:
        return {}
    qs = PageImage.objects.filter(page_id__in=all_page_ids).select_related("page")
    grouped = defaultdict(list)
    for img in qs:
        if img.file:
            grouped[img.page_id].append(request.build_absolute_uri(img.file.url))
    # keep insertion order per page
    return {pid: urls for pid, urls in grouped.items()}

def _role_of_langchain_msg(msg) -> str:
    t = getattr(msg, "type", "").lower()
    if t == "human": return "user"
    if t == "ai":    return "assistant"
    if t == "tool":  return "tool"
    if t == "system":return "system"
    return t or msg.__class__.__name__.replace("Message", "").lower()

def _content_to_raw_str(raw) -> str:
    # LangChain message content can be str or list of typed parts
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw, ensure_ascii=False)
    except Exception:
        return str(raw)

def _annotate_turn_ids_inplace(history_ua: list[dict]) -> None:
    """
    In-place: add turn_id to each user/assistant message.
    Pairs in order: (user -> assistant) => 1..N
    """
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

def _build_turn_pairs(history_ua: list[dict]) -> list[dict]:
    """
    Build convenient [{id, user, assistant, images, sources}] list.
    Requires messages to already have turn_id and assistant messages to carry images/sources.
    """
    pairs = []
    cur = {}
    for m in history_ua:
        if m["role"] == "user":
            cur = {"id": m.get("turn_id"), "user": m.get("content", "")}
        elif m["role"] == "assistant":
            cur = {"id": m.get("turn_id"), **cur}
            cur["assistant"] = m.get("content", "")
            cur["images"] = m.get("images", []) or []
            cur["sources"] = m.get("sources", []) or []
            pairs.append(cur)
            cur = {}
    return pairs

# ──────────────────────────────────────────────────────────────────────────────
# POST /api/chat/  →  run a turn, return current answer + complete history
# ──────────────────────────────────────────────────────────────────────────────

@method_decorator(csrf_exempt, name="dispatch")
class ChatAPIView(APIView):
    def post(self, request):
        user_text = request.data.get("message")
        if not user_text:
            return Response({"error": "Field 'message' is required."},
                            status=status.HTTP_400_BAD_REQUEST)

        thread_id_in = request.data.get("thread_id")
        try:
            # Run the turn (gives final answer + urls for THIS assistant turn)
            answer, urls_current, thread_id_out, _ = run_turn(user_text, thread_id=thread_id_in)

            # After the turn, read RAW messages from the LangGraph memory
            from .agent import get_thread_messages  # local import to avoid cycles
            msgs = get_thread_messages(thread_id_out) or []
        except Exception as exc:
            logger.exception("Chatbot crashed")
            if settings.DEBUG:
                return Response(
                    {"error": str(exc), "type": exc.__class__.__name__, "traceback": traceback.format_exc()},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response({"error": "Internal Server Error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # 1) Linearize raw messages (keep tool/system in this pass)
        linear = []
        for m in msgs:
            role = _role_of_langchain_msg(m)
            linear.append({"role": role, "content_raw": _content_to_raw_str(m.content)})

        # 2) For every assistant message, bind to its own tool (bounded) and collect page_ids/urls
        assistant_indices = [i for i, r in enumerate(linear) if linear[i]["role"] == "assistant"]
        per_assistant_page_ids: dict[int, list[int]] = {}
        per_assistant_sources: dict[int, list[str]] = {}
        all_page_ids: list[int] = []

        for ai in assistant_indices:
            # Parse assistant JSON for display text + assistant-declared urls
            content_disp = linear[ai]["content_raw"]
            urls_from_assistant = []
            try:
                payload = json.loads(content_disp)
                if isinstance(payload, dict) and "response" in payload:
                    content_disp = payload.get("response", content_disp)
                    urls_from_assistant = payload.get("urls", []) or []
            except Exception:
                pass
            # Save back the display text
            linear[ai]["content_disp"] = content_disp

            # Find THIS turn's tool (bounded by previous assistant)
            pids, urls_from_tool = _nearest_tool_before_bounded(linear, assistant_idx=ai)
            per_assistant_page_ids[ai] = pids
            all_page_ids.extend(pids)

            # Prefer assistant-declared urls, else fallback to tool urls
            per_assistant_sources[ai] = urls_from_assistant if urls_from_assistant else urls_from_tool

        # 3) Batch-fetch all images for all assistant turns
        all_page_ids = list(dict.fromkeys(all_page_ids))
        page_images_map = _build_page_images_map(all_page_ids, request)

        # 4) Build user/assistant-only history with images/sources attached to each assistant
        history_ua = []
        for i, rec in enumerate(linear):
            role = rec["role"]
            if role == "tool" or role == "system":
                continue
            if role == "user":
                history_ua.append({"role": "user", "content": rec["content_raw"]})
            elif role == "assistant":
                # Flatten images for this assistant, ordered by page_ids
                pids = per_assistant_page_ids.get(i, []) or []
                imgs = []
                for pid in pids:
                    imgs.extend(page_images_map.get(pid, []))
                imgs = list(dict.fromkeys(imgs))

                history_ua.append({
                    "role": "assistant",
                    "content": rec.get("content_disp", rec["content_raw"]),
                    "images": imgs,
                    "sources": per_assistant_sources.get(i, []) or [],
                })

        # 5) Annotate turn ids and build pairs
        _annotate_turn_ids_inplace(history_ua)
        turns = _build_turn_pairs(history_ua)

        # 6) Top-level images: ONLY for the current turn (last assistant)
        current_ai = assistant_indices[-1] if assistant_indices else None
        current_images_by_page = {}
        if current_ai is not None:
            current_pids = per_assistant_page_ids.get(current_ai, []) or []
            current_images_by_page = {pid: page_images_map.get(pid, []) for pid in current_pids if page_images_map.get(pid)}

        return Response({
            "thread_id": thread_id_out,
            "answer": answer,
            "history": history_ua,
        })

# ──────────────────────────────────────────────────────────────────────────────
# GET /api/chat/<thread_id>/  →  return full history (images/sources per turn)
# ──────────────────────────────────────────────────────────────────────────────

@method_decorator(csrf_exempt, name="dispatch")
class ChatHistoryAPIView(APIView):
    def get(self, request, thread_id: str):
        from .agent import get_thread_messages  # local import to avoid cycles
        msgs = get_thread_messages(thread_id) or []
        if not msgs:
            return Response({"thread_id": thread_id, "history": [], "turns": []}, status=status.HTTP_200_OK)

        # 1) Linearize raw messages (keep tool/system for bounded lookup)
        linear = []
        for m in msgs:
            role = _role_of_langchain_msg(m)
            linear.append({"role": role, "content_raw": _content_to_raw_str(m.content)})

        # 2) Collect per-assistant page_ids & sources
        assistant_indices = [i for i, r in enumerate(linear) if linear[i]["role"] == "assistant"]
        per_assistant_page_ids: dict[int, list[int]] = {}
        per_assistant_sources: dict[int, list[str]] = {}
        all_page_ids: list[int] = []

        for ai in assistant_indices:
            # From assistant JSON
            content_disp = linear[ai]["content_raw"]
            urls_from_assistant = []
            try:
                payload = json.loads(content_disp)
                if isinstance(payload, dict) and "response" in payload:
                    content_disp = payload.get("response", content_disp)
                    urls_from_assistant = payload.get("urls", []) or []
            except Exception:
                pass
            linear[ai]["content_disp"] = content_disp

            # From bounded tool
            pids, urls_from_tool = _nearest_tool_before_bounded(linear, assistant_idx=ai)
            per_assistant_page_ids[ai] = pids
            all_page_ids.extend(pids)

            # Prefer assistant urls; else tool urls
            per_assistant_sources[ai] = urls_from_assistant if urls_from_assistant else urls_from_tool

        # 3) Batch images
        all_page_ids = list(dict.fromkeys(all_page_ids))
        page_images_map = _build_page_images_map(all_page_ids, request)

        # 4) Build user/assistant-only history with images/sources per assistant
        history_ua = []
        for i, rec in enumerate(linear):
            role = rec["role"]
            if role == "tool" or role == "system":
                continue
            if role == "user":
                history_ua.append({"role": "user", "content": rec["content_raw"]})
            elif role == "assistant":
                pids = per_assistant_page_ids.get(i, []) or []
                imgs = []
                for pid in pids:
                    imgs.extend(page_images_map.get(pid, []))
                imgs = list(dict.fromkeys(imgs))
                history_ua.append({
                    "role": "assistant",
                    "content": rec.get("content_disp", rec["content_raw"]),
                    "images": imgs,
                    "sources": per_assistant_sources.get(i, []) or [],
                })

        # 5) Turn ids + pairs
        _annotate_turn_ids_inplace(history_ua)
        turns = _build_turn_pairs(history_ua)

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