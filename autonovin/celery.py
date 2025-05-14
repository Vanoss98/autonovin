from celery import Celery
import os


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'autonovin.settings')
app = Celery("autonovin")
app.config_from_object('django.conf:settings', namespace="CELERY")
app.autodiscover_tasks(['ingestion', 'crawler', 'retrieval'])
