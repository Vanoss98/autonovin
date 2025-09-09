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
    # Look at the most recent tool message first
    for msg in reversed(history):
        if msg.get("role") == "tool":
            try:
                payload = json.loads(msg.get("content") or "{}")
                if "page_ids" in payload:
                    return payload["page_ids"]
                # Fallback: try to infer from docs if tool wasn't updated everywhere yet
                if "docs" in payload:
                    inferred = []
                    for _, lst in (payload["docs"] or {}).items():
                        for tup in lst:
                            if isinstance(tup, (list, tuple)) and len(tup) >= 9 and tup[8] is not None:
                                inferred.append(tup[8])
                    if inferred:
                        # preserve order & dedupe
                        return [pid for pid in dict.fromkeys(inferred)]
            except Exception:
                pass
    return []


@method_decorator(csrf_exempt, name="dispatch")
class ChatAPIView(APIView):
    def post(self, request):
        user_text = request.data.get("message")
        if not user_text:
            return Response({"error": "Field 'message' is required."},
                            status=status.HTTP_400_BAD_REQUEST)

        thread_id_in = request.data.get("thread_id")
        try:
            answer, urls, thread_id_out, history = run_turn(user_text, thread_id=thread_id_in)
        except Exception as exc:
            logger.exception("Chatbot crashed")
            if settings.DEBUG:
                return Response(
                    {"error": str(exc), "type": exc.__class__.__name__, "traceback": traceback.format_exc()},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response({"error": "Internal Server Error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # NEW: fetch all images for all retrieved page_ids
        page_ids = _extract_page_ids_from_history(history)
        images_by_page: dict[int, list[dict]] = {}
        if page_ids:
            qs = PageImage.objects.filter(page_id__in=page_ids).select_related("page")
            grouped = defaultdict(list)
            for img in qs:
                grouped[img.page_id].append(request.build_absolute_uri(img.file.url) if img.file else None)
            images_by_page = dict(grouped)

        return Response({
            "thread_id": thread_id_out,
            "answer": answer,
            "urls": urls,
            "images": images_by_page,
            "history": history,
        })
