from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .agent import run_turn
import logging, traceback
from django.conf import settings

logger = logging.getLogger(__name__)


class ChatAPIView(APIView):
    def post(self, request):
        user_text = request.data.get("message")
        if not user_text:
            return Response(
                {"error": "Field 'message' is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        thread_id_in = request.data.get("thread_id")       # may be None
        try:
            answer, urls, thread_id_out = run_turn(user_text, thread_id=thread_id_in)
        except Exception as exc:
            # 1) log the full traceback to console / file
            logger.exception("Chatbot crashed")

            # 2) in dev, send the traceback back to the client as well
            if settings.DEBUG:
                return Response(
                    {
                        "error": str(exc),
                        "type": exc.__class__.__name__,
                        "traceback": traceback.format_exc(),
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            # 3) in prod, keep the response minimal
            return Response(
                {"error": "Internal Server Error"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {"thread_id": thread_id_out, "answer": answer, "urls": urls}
        )
