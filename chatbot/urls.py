from django.urls import path
from .views import ChatAPIView, ChatDebugAPIView

urlpatterns = [
    path("chat/", ChatAPIView.as_view(), name="chat"),
    path("api/chat/debug/", ChatDebugAPIView.as_view(), name="chat-debug"),
    # path("get-history/<str:thread_id>", ChatHistoryAPIView.as_view(), name="get-history")
]
