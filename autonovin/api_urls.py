from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("ingest/", include('ingestion.urls')),
    path("crawl/", include('crawler.urls')),
    path("chatbot/", include('chatbot.urls')),
    path("ocr/", include('ocr.urls')),
    path("compliance/", include('compliance.urls')),
]
