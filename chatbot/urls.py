from django.urls import path
from .views import ChatAPIView, ChatHistoryAPIView

urlpatterns = [
    path("chat/", ChatAPIView.as_view(), name="chat"),
    path("get-history/<str:thread_id>", ChatHistoryAPIView.as_view(), name="get-history")
]
