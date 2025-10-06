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

# ───────────────────────────────────────────────────────────────────────────────
# Helpers (بدون تغییر اساسی؛ فقط تمیزتر و امن‌تر)
# ───────────────────────────────────────────────────────────────────────────────
def _content_to_raw_str(raw) -> str:
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw, ensure_ascii=False)
    except Exception:
        return str(raw)

def _role_of_langchain_msg(msg) -> str:
    t = getattr(msg, "type", "").lower()
    if t == "human": return "user"
    if t == "ai":    return "assistant"
    if t == "tool":  return "tool"
    if t == "system":return "system"
    return t or msg.__class__.__name__.replace("Message", "").lower()

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
        if role == "assistant":
            return [], []
        if role == "tool":
            return _parse_tool_payload(full_linear[j].get("content_raw"))
        j -= 1
    return [], []

def _build_page_images_map(all_page_ids: list[int], request):
    if not all_page_ids:
        return {}
    qs = PageImage.objects.filter(page_id__in=all_page_ids).select_related("page")
    grouped = defaultdict(list)
    for img in qs:
        if img.file:
            grouped[img.page_id].append(request.build_absolute_uri(img.file.url))
    return {pid: urls for pid, urls in grouped.items()}

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

def _build_turn_pairs(history_ua: list[dict]) -> list[dict]:
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

# ───────────────────────────────────────────────────────────────────────────────
# POST /api/chat/  →  یک نوبت چت؛ سریع‌تر و با تصاویر منبع
# ───────────────────────────────────────────────────────────────────────────────
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
            msgs = get_thread_messages(thread_id_out) or []
        except Exception as exc:
            logger.exception("Chatbot crashed")
            if settings.DEBUG:
                return Response(
                    {"error": str(exc), "type": exc.__class__.__name__, "traceback": traceback.format_exc()},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response({"error": "Internal Server Error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # 1) Linearize (با tool/system برای بایند مرجع)
        linear = []
        for m in msgs:
            linear.append({"role": _role_of_langchain_msg(m), "content_raw": _content_to_raw_str(m.content)})

        # 2) برای هر پاسخ Assistant، صفحات و منابع همان نوبت را پیدا کن
        assistant_indices = [i for i, rec in enumerate(linear) if rec["role"] == "assistant"]
        per_ai_page_ids, per_ai_sources, all_page_ids = {}, {}, []

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

        # 3) تصاویر همهٔ صفحات (یک‌باره)
        all_page_ids = list(dict.fromkeys(all_page_ids))
        page_images_map = _build_page_images_map(all_page_ids, request)

        # 4) تاریخچهٔ user/assistant با الحاق تصاویر و منابع
        history_ua = []
        for i, rec in enumerate(linear):
            role = rec["role"]
            if role in ("tool","system"):
                continue
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
