# compliance/bootsteps.py
import os
import json
import logging
from kombu import Queue, Exchange, Consumer
from celery import bootsteps
from compliance.tasks import upsert_ad_vector  # celery task

log = logging.getLogger(__name__)

# Read from env (provide sane defaults)
QUEUE_SELL = os.getenv("QUEUE_SELL", "car_ads_sell")
QUEUE_BUY  = os.getenv("QUEUE_BUY",  "car_ads_buy")

RAW_QUEUE_SELL = Queue(
    name=QUEUE_SELL,
    exchange=Exchange("", type="direct"),  # default (empty) exchange
    routing_key=QUEUE_SELL,
    durable=True,
)

RAW_QUEUE_BUY = Queue(
    name=QUEUE_BUY,
    exchange=Exchange("", type="direct"),
    routing_key=QUEUE_BUY,
    durable=True,
)

class RawAdsConsumer(bootsteps.ConsumerStep):
    """
    Consume SELL and BUY ads from separate queues and forward to the Celery task.
    """

    def get_consumers(self, channel):
        return [
            Consumer(
                channel,
                queues=[RAW_QUEUE_SELL, RAW_QUEUE_BUY],
                callbacks=[self.handle],
                accept=["json"],
                prefetch_count=50,
            )
        ]

    def handle(self, body, message):
        try:
            ad_json = json.dumps(body)  # body already a dict
            upsert_ad_vector.delay(ad_json)
            mid = (body.get("message") or {}).get("id")
            qname = message.delivery_info.get("routing_key")
            log.info("queued ad %s from %s", mid if mid is not None else "(no id)", qname)
            message.ack()
        except Exception:
            log.exception("error processing ad")
            message.reject(requeue=True)
