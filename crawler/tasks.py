from celery import shared_task
from asgiref.sync import async_to_sync
from .service import CrawlService, ImageDownloadService


@shared_task()
def execute_crawling(config):
    """ Task for crawling """
    service = CrawlService()
    async_to_sync(service.crawl)(config)


@shared_task(bind=True, name="ingest.download_images")
def download_images(self, page_id: int):
    service = ImageDownloadService()
    new_count = async_to_sync(service.run)(page_id)
    return {"saved": new_count}
