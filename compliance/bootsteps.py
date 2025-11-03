# compliance/bootsteps.py
# -*- coding: utf-8 -*-
import json
import logging
from kombu import Queue, Consumer
from celery import bootsteps
from django.conf import settings
from compliance.tasks import upsert_ad_vector

print("[BOOTSTEP] importing compliance.bootsteps ...")
log = logging.getLogger(__name__)

Q_SELL = getattr(settings, "COMPLIANCE_QUEUE_SELL", None)
Q_BUY  = getattr(settings, "COMPLIANCE_QUEUE_BUY", None)
print(f"[BOOTSTEP] settings queues → SELL={Q_SELL!r}  BUY={Q_BUY!r}")

RAW_QUEUE_SELL = Queue(name=Q_SELL, durable=True, no_declare=True) if Q_SELL else None
RAW_QUEUE_BUY  = Queue(name=Q_BUY,  durable=True, no_declare=True) if Q_BUY else None

class RawAdsConsumer(bootsteps.ConsumerStep):
    def __init__(self, *a, **kw):
        print("[BOOTSTEP] RawAdsConsumer.__init__")
        super().__init__(*a, **kw)

    def get_consumers(self, channel):
        print("[BOOTSTEP] get_consumers called; building kombu.Consumer ...")
        queues = []
        if RAW_QUEUE_SELL: queues.append(RAW_QUEUE_SELL)
        if RAW_QUEUE_BUY:  queues.append(RAW_QUEUE_BUY)
        print(f"[BOOTSTEP] Consumers will attach to queues: {[q.name for q in queues]}")

        return [
            Consumer(
                channel,
                queues=queues,
                callbacks=[self.handle],
                # no 'accept' → accept any content-type; we'll parse manually
            )
        ]

    def handle(self, body, message):
        qname = message.delivery_info.get("routing_key")
        ct = getattr(message, "content_type", None)
        print(f"[BOOTSTEP] received delivery | queue={qname} content_type={ct} type={type(body)}")
        try:
            # preview body
            if isinstance(body, (bytes, bytearray, memoryview)):
                raw = bytes(body).decode("utf-8", errors="ignore")
                print(f"[BOOTSTEP] raw bytes preview: {raw[:300]}{'...<trunc>' if len(raw)>300 else ''}")
                try:
                    payload = json.loads(raw)
                except Exception as e:
                    print(f"[BOOTSTEP] ERROR json.loads(bytes): {e!r}")
                    payload = None
            elif isinstance(body, str):
                print(f"[BOOTSTEP] raw str preview: {body[:300]}{'...<trunc>' if len(body)>300 else ''}")
                try:
                    payload = json.loads(body)
                except Exception as e:
                    print(f"[BOOTSTEP] ERROR json.loads(str): {e!r}")
                    payload = None
            elif isinstance(body, dict):
                payload = body
                print(f"[BOOTSTEP] dict envelope keys: {list(payload.keys())[:10]}")
            else:
                print(f"[BOOTSTEP] Unexpected body type: {type(body)}")
                payload = None

            if not isinstance(payload, dict) or "message" not in payload:
                print("[BOOTSTEP] Skipping: not an MT envelope (no 'message' key). ACK and move on.")
                message.ack()
                return

            # forward to Celery task
            ad_json = json.dumps(payload, ensure_ascii=False)
            print(f"[BOOTSTEP] forwarding to task compliance.upsert_ad_vector; bytes={len(ad_json.encode('utf-8'))}")
            upsert_ad_vector.delay(ad_json)
            print(f"[BOOTSTEP] forwarded ✓ (id={payload.get('message',{}).get('id')})")
            message.ack()

        except Exception as e:
            print(f"[BOOTSTEP] ERROR processing message: {e!r}")
            log.exception("Error processing MQ message")
            message.reject(requeue=True)
