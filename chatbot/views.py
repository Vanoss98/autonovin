# views.py
import json, logging, traceback
from collections import defaultdict

from django.conf import settings
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from crawler.infrastructure.models import PageImage
from .agent import run_turn, get_thread_messages

logger = logging.getLogger(__name__)

# ───────── Helpers ─────────
def _content_to_raw_str(raw) -> str:
    if isinstance(raw, str): return raw
    try: return json.dumps(raw, ensure_ascii=False)
    except Exception: return str(raw)

def _role_of_langchain_msg(msg) -> str:
    t = getattr(msg, "type", "").lower()
    if t == "human": return "user"
    if t == "ai":    return "assistant"
    if t == "tool":  return "tool"
    if t == "system":return "system"
    return t or msg.__class__.__name__.replace("Message","").lower()

def _parse_tool_payload(raw: str):
    page_ids, urls = [], []
    try:
        payload = json.loads(raw or "{}")
        if isinstance(payload, dict):
            if isinstance(payload.get("urls"), list):
                urls = [u for u in payload["urls"] if isinstance(u, str)]
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

def _nearest_tool_before_bounded(full_linear: list[dict], *, assistant_idx: int):
    j = assistant_idx - 1
    while j >= 0:
        role = full_linear[j].get("role")
        if role == "assistant": return [], []
        if role == "tool":      return _parse_tool_payload(full_linear[j].get("content_raw"))
        j -= 1
    return [], []

def _build_page_images_map(all_page_ids: list[int], request):
    if not all_page_ids: return {}
    qs = PageImage.objects.filter(page_id__in=all_page_ids).select_related("page")
    grouped = defaultdict(list)
    for img in qs:
        if img.file:
            grouped[img.page_id].append(request.build_absolute_uri(img.file.url))
    return {pid: urls for pid, urls in grouped.items()}

def _annotate_turn_ids_inplace(history_ua: list[dict]) -> None:
    turn_id, pending_user = 0, None
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

# ───────── POST /api/chat/ ─────────
@method_decorator(csrf_exempt, name="dispatch")
class ChatAPIView(APIView):
    def post(self, request):
        user_text = request.data.get("message")
        if not user_text:
            return Response({"error": "Field 'message' is required."}, status=status.HTTP_400_BAD_REQUEST)

        thread_id_in = request.data.get("thread_id")
        try:
            answer, urls_current, thread_id_out, _ = run_turn(user_text, thread_id=thread_id_in)
            msgs = get_thread_messages(thread_id_out) or []
        except Exception as exc:
            logger.exception("Chatbot crashed")
            if settings.DEBUG:
                return Response(
                    {"error": str(exc), "type": exc.__class__.__name__, "traceback": traceback.format_exc()},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response({"error": "Internal Server Error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # 1) خطی‌سازی
        linear = [{"role": _role_of_langchain_msg(m), "content_raw": _content_to_raw_str(m.content)} for m in msgs]

        # 2) اتصال هر پاسخ Assistant به ابزار همان نوبت
        assistant_indices = [i for i, rec in enumerate(linear) if rec["role"] == "assistant"]
        per_ai_page_ids, per_ai_sources, all_page_ids = {}, {}, []
        history_ua = []

        for ai in assistant_indices:
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

            pids, urls_from_tool = _nearest_tool_before_bounded(linear, assistant_idx=ai)
            per_ai_page_ids[ai] = pids
            all_page_ids.extend(pids)
            per_ai_sources[ai] = urls_from_assistant if urls_from_assistant else urls_from_tool

        # 3) تصاویر همهٔ صفحات
        all_page_ids = list(dict.fromkeys(all_page_ids))
        page_images_map = _build_page_images_map(all_page_ids, request)

        # 4) تاریخچهٔ user/assistant با تصاویر/منابع
        for i, rec in enumerate(linear):
            role = rec["role"]
            if role in ("tool","system"): continue
            if role == "user":
                history_ua.append({"role": "user", "content": rec["content_raw"]})
            else:
                pids = per_ai_page_ids.get(i, []) or []
                imgs = []
                for pid in pids:
                    imgs.extend(page_images_map.get(pid, []))
                imgs = list(dict.fromkeys(imgs))
                history_ua.append({
                    "role": "assistant",
                    "content": rec.get("content_disp", rec["content_raw"]),
                    "images": imgs,
                    "sources": per_ai_sources.get(i, []) or [],
                })

        _annotate_turn_ids_inplace(history_ua)
        return Response({
            "thread_id": thread_id_out,
            "answer": answer,
            "history": history_ua,
        })


import json, logging, traceback, time
from collections import defaultdict

from django.conf import settings
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from crawler.infrastructure.models import PageImage
from .agent import run_turn, get_thread_messages
from .utils import analyze_query, hybrid_search, gen_answer_prompt, generate_answer  # ← اضافه


@method_decorator(csrf_exempt, name="dispatch")
class ChatDebugAPIView(APIView):
    """
    Request (POST):
      { "message": "..." , "top_k": 15 }
    Response:
      {
        "thread_id": null,                  # این اندپوینت بدون گراف کار می‌کند
        "analyzer": {...},                  # خروجی analyze_query (JSON)
        "retrieval": {
          "k": 15,
          "by_model": {
            "BMW X5": [
              {"id": "...", "title": "...", "score": 1.234, "url": "...", "page_id": 123,
               "model_id": "bmw_x5", "images": ["..."], "snippet": "اولین 280 کاراکتر..."}
            ],
            "ALL": [...]
          },
          "total_docs": 37
        },
        "llm_answer": {
          "raw": "پاسخ LLM یا پیام عدم وجود اطلاعات",
          "sources": ["url1","url2"]
        },
        "timings_ms": {"analyze": 0, "retrieve": 0, "llm": 0, "total": 0}
      }
    """
    def post(self, request):
        user_text = request.data.get("message")
        top_k = int(request.data.get("top_k") or 15)
        if not user_text:
            return Response({"error": "Field 'message' is required."},
                            status=status.HTTP_400_BAD_REQUEST)

        t0 = time.perf_counter()
        try:
            # 1) Analyzer (LLM همان gpt-4.1 با JSON mode در utils.analyze_query)
            t1 = time.perf_counter()
            analysis = analyze_query(user_text) or {}
            t2 = time.perf_counter()

            # 2) Retrieval (Hybrid: Chroma + BM25 روی کاندیداها)
            docs_by_model = hybrid_search(analysis, top_k=top_k) or {}
            t3 = time.perf_counter()

            # ساخت خروجی فشردهٔ رتریوال برای دیباگ (بدون محتواهای سنگین)
            total_docs = sum(len(v or []) for v in docs_by_model.values())
            retrieval_slim = {}
            for model_key, pack in (docs_by_model or {}).items():
                slim = []
                for tup in (pack or []):
                    # tup = (id, title, content, score, images, model_id, url, source, page_id)
                    cid, title, content, score, imgs, model_id, url, _, page_id = tup
                    slim.append({
                        "id": cid,
                        "title": title,
                        "score": float(score),
                        "url": url,
                        "page_id": page_id,
                        "model_id": model_id,
                        "images": imgs[:3],
                        "snippet": (content or "")[:280]
                    })
                retrieval_slim[model_key] = slim

            # 3) LLM Answer (فقط اگر سند مرتبط داریم؛ وگرنه پیام نداشتن دیتا)
            if any(docs_by_model.values()):
                # اسناد را به یک لیست تخت برای پرامپت پاسخ تبدیل کنیم
                merged_docs = []
                for pack in docs_by_model.values():
                    merged_docs.extend(pack)
                merged_docs = merged_docs[:top_k]  # پرهیز از پرامپت خیلی بزرگ

                # generate_answer همان gpt-4.1 را صدا می‌زند
                answer_raw, _, _ = generate_answer(user_text, merged_docs)
                llm_payload = json.loads(answer_raw) if isinstance(answer_raw, str) else answer_raw
                if not isinstance(llm_payload, dict) or "response" not in llm_payload:
                    # اگر مدل JSON برنگرداند، متن خام را هم نمایش بدهیم
                    llm_out = {"raw": answer_raw, "sources": []}
                else:
                    llm_out = {
                        "raw": llm_payload.get("response",""),
                        "sources": llm_payload.get("sources",[]) or []
                    }
            else:
                llm_out = {
                    "raw": "متأسفانه اطلاعات درخواستی در مستندات ارائه‌شده موجود نیست.",
                    "sources": []
                }
            t4 = time.perf_counter()

            return Response({
                "thread_id": None,
                "analyzer": analysis,
                "retrieval": {
                    "k": top_k,
                    "by_model": retrieval_slim,
                    "total_docs": total_docs
                },
                "llm_answer": llm_out,
                "timings_ms": {
                    "analyze": int((t2 - t1) * 1000),
                    "retrieve": int((t3 - t2) * 1000),
                    "llm": int((t4 - t3) * 1000),
                    "total": int((t4 - t0) * 1000),
                }
            }, status=status.HTTP_200_OK)

        except Exception as exc:
            logger.exception("ChatDebug crashed")
            if settings.DEBUG:
                return Response(
                    {"error": str(exc), "type": exc.__class__.__name__,
                     "traceback": traceback.format_exc()},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response({"error": "Internal Server Error"},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)