import json, logging
from kombu import Queue, Exchange, Consumer
from celery import bootsteps
from compliance.tasks import upsert_ad_vector   # Celery task

log = logging.getLogger(__name__)

RAW_QUEUE = Queue(
    name="car_ads",
    exchange=Exchange("", type="direct"),   # default no-exchange
    routing_key="car_ads",
    durable=True,
)

class RawAdsConsumer(bootsteps.ConsumerStep):
    """
    Streams plain JSON messages from RabbitMQ → Celery task.
    """

    def get_consumers(self, channel):
        return [
            Consumer(
                channel,
                queues=[RAW_QUEUE],
                callbacks=[self.handle],
                accept=["json"],
                prefetch_count=50,           # tweak for throughput
            )
        ]

    def handle(self, body, message):
        try:
            ad_json = json.dumps(body)          # body already dict
            upsert_ad_vector.delay(ad_json)     # enqueue task
            log.info("queued ad %s", body.get("id"))
            message.ack()
        except Exception:
            log.exception("error processing ad")
            message.reject(requeue=True)
