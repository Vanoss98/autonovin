# autonovin/celery.py
import os
import logging
from celery import Celery

print("[CELERY] importing autonovin.celery ...")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "autonovin.settings")

app = Celery("autonovin")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
print("[CELERY] Celery app created & autodiscover_tasks() called")

log = logging.getLogger(__name__)

@app.on_after_configure.connect
def _register_bootsteps(sender, **kwargs):
    print("[CELERY] on_after_configure fired; registering RawAdsConsumer bootstep ...")
    try:
        from compliance.bootsteps import RawAdsConsumer  # late import
        sender.steps["consumer"].add(RawAdsConsumer)
        print("[CELERY] RawAdsConsumer registered ✓")
        log.info("Registered RawAdsConsumer bootstep.")
    except Exception as exc:
        print(f"[CELERY] ERROR registering RawAdsConsumer: {exc!r}")
        log.exception("Failed to register RawAdsConsumer: %s", exc)

@app.task(bind=True)
def debug_task(self):
    print(f"[CELERY] debug_task Request: {self.request!r}")
