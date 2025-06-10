from django.urls import path
from .views import ReplacePlateAPIView

urlpatterns = [
    path("replace-plate/", ReplacePlateAPIView.as_view(), name="replace-plate"),
]
