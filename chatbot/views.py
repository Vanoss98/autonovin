# views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .agent import run_turn
import logging, traceback
from django.conf import settings
from collections import defaultdict
from crawler.infrastructure.models import PageImage
import json
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

logger = logging.getLogger(__name__)

def _extract_page_ids_from_history(history: list[dict]) -> list[int]:
    # Unchanged: rely on the tool message to pull page_ids
    for msg in reversed(history):
        if msg.get("role") == "tool":
            try:
                payload = json.loads(msg.get("content") or "{}")
                if "page_ids" in payload:
                    return payload["page_ids"]
                if "docs" in payload:
                    inferred = []
                    for _, lst in (payload["docs"] or {}).items():
                        for tup in lst:
                            if isinstance(tup, (list, tuple)) and len(tup) >= 9 and tup[8] is not None:
                                inferred.append(tup[8])
                    if inferred:
                        return [pid for pid in dict.fromkeys(inferred)]
            except Exception:
                pass
    return []

def _flatten_images(images_by_page: dict[int, list[str]]) -> list[str]:
    out = []
    for urls in images_by_page.values():
        for u in urls:
            if u:
                out.append(u)
    # de-dupe while preserving order
    return list(dict.fromkeys(out))

def _filter_user_assistant(history: list[dict]) -> list[dict]:
    # Only user & assistant roles for client
    return [m for m in history if m.get("role") in ("user", "assistant")]

def _annotate_turn_ids(history_ua: list[dict]) -> None:
    """
    In-place: add turn_id to each user/assistant message.
    Pairs in order: (user -> assistant) => id = 1,2,3,...
    If history starts with assistant (rare), it will be paired with previous user-less turn.
    """
    turn_id = 0
    pending_user = None
    for m in history_ua:
        r = m.get("role")
        if r == "user":
            pending_user = m
        elif r == "assistant":
            # Close a pair when we see an assistant.
            turn_id += 1
            if pending_user is not None and "turn_id" not in pending_user:
                pending_user["turn_id"] = turn_id
            m["turn_id"] = turn_id
            pending_user = None

def _build_turn_pairs(history_ua: list[dict]) -> list[dict]:
    """
    Build convenient [{id, user, assistant, images, sources}] list.
    Reads turn_id from messages (must run _annotate_turn_ids first).
    """
    pairs = []
    cur = {}
    for m in history_ua:
        if m["role"] == "user":
            cur = {"id": m.get("turn_id"), "user": m.get("content", "")}
        elif m["role"] == "assistant":
            # finalize a pair
            cur = {"id": m.get("turn_id"), **cur}
            cur["assistant"] = m.get("content", "")
            cur["images"] = m.get("images", []) or []
            cur["sources"] = m.get("sources", []) or []
            pairs.append(cur)
            cur = {}
    return pairs

@method_decorator(csrf_exempt, name="dispatch")
class ChatAPIView(APIView):
    def post(self, request):
        user_text = request.data.get("message")
        if not user_text:
            return Response({"error": "Field 'message' is required."},
                            status=status.HTTP_400_BAD_REQUEST)

        thread_id_in = request.data.get("thread_id")
        try:
            # answer = assistant text, urls = sources used, history = full history including tool
            answer, urls, thread_id_out, history = run_turn(user_text, thread_id=thread_id_in)
        except Exception as exc:
            logger.exception("Chatbot crashed")
            if settings.DEBUG:
                return Response(
                    {"error": str(exc), "type": exc.__class__.__name__, "traceback": traceback.format_exc()},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response({"error": "Internal Server Error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # 1) Fetch page images for THIS turn (via tool's page_ids)
        page_ids = _extract_page_ids_from_history(history)
        images_by_page: dict[int, list[str]] = {}
        if page_ids:
            qs = PageImage.objects.filter(page_id__in=page_ids).select_related("page")
            grouped = defaultdict(list)
            for img in qs:
                grouped[img.page_id].append(request.build_absolute_uri(img.file.url) if img.file else None)
            images_by_page = dict(grouped)

        # 2) Flatten to a single images[] for this assistant turn
        images_flat = _flatten_images(images_by_page)

        # 3) Attach images + sources onto the **last assistant** message in history (in-place)
        last_assistant_idx = None
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("role") == "assistant":
                last_assistant_idx = i
                break
        if last_assistant_idx is not None:
            history[last_assistant_idx]["images"] = images_flat
            history[last_assistant_idx]["sources"] = urls

        # 4) Prepare a user/assistant-only version + annotate turn_ids
        history_ua = _filter_user_assistant(history)
        _annotate_turn_ids(history_ua)

        # 5) Build a friendly paired structure (useful for UIs)
        turns = _build_turn_pairs(history_ua)

        # Optional: if you want “no tool messages” at all, return `history_ua`;
        # if you prefer backward compatibility, return both.
        return Response({
            "thread_id": thread_id_out,
            "answer": answer,
            "urls": urls,
            "images": images_by_page,
            "history": history_ua,
            "turns": turns
        })
