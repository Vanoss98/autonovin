from django.urls import path
from .views import *

urlpatterns = [
    path("related-ads/", RetrieveAdsByBrandModelView.as_view(), name="related-ads"),
    path("queue-peek/", queue_peek, name="queue-peek"),
    path("chroma-dump/", chroma_dump, name="chroma_dump"),
    path("seed-sample-ads/", seed_sample_ads, name="seed_sample_ads"),
]
