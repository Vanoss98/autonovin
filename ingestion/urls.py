from django.urls import path
from .views import Embedder, List, Retrieve

urlpatterns = [
    path("embed/<int:pk>", Embedder.as_view(), name="embedder"),
    path("list/", List.as_view(), name="list"),
    path("retrieve/", Retrieve.as_view(), name="retrieve"),
]
