from django.urls import path
from .views import *

urlpatterns = [
    path("related-ads/", RetrieveAdsByBrandModelView.as_view(), name="related-ads"),
    # path("queue-peek/", queue_peek, name="queue-peek"),
]
