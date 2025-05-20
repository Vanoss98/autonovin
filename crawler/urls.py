from django.urls import path
from .views import CrawlExecutionView


urlpatterns = [
    path('execute/', CrawlExecutionView.as_view(), name='crawl-execution')
]
