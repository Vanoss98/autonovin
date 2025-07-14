import os
from celery import Celery
import compliance.bootsteps
from compliance.bootsteps import RawAdsConsumer

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "autonovin.settings")
app = Celery("autonovin")
app.config_from_object("django.conf:settings", namespace="CELERY")

# autodiscover tasks in both crawler and compliance apps
app.autodiscover_tasks(["crawler", "compliance"])

# boot-step consumer

app.steps["consumer"].add(RawAdsConsumer)
