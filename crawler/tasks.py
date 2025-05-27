from celery import shared_task, chain
from asgiref.sync import async_to_sync
from .application.services.crawl4ai_crawler_service import Crawl4aiCrawlerService
from .application.services.image_download_service import ImageDownloadService
from .application.use_cases.page_crawler import PageCrawlerUseCase
from .infrastructure.repositories.page_repository import PageRepository
from .infrastructure.repositories.page_image_repository import PageImageRepository

# Initialize repositories and services
page_repo = PageRepository()
image_repo = PageImageRepository()
crawler_service = Crawl4aiCrawlerService()
image_downloader = ImageDownloadService()


# Wrap the PageCrawlerUseCase into a Celery task
@shared_task(bind=True)
def page_crawler_task(self, config):
    page_crawler_use_case = PageCrawlerUseCase(crawler=crawler_service, repository=page_repo,
                                               image_downloader=image_downloader, image_repository=image_repo)
    # Convert the async method to sync using async_to_sync
    page_ids = async_to_sync(page_crawler_use_case.execute)(config)
    return page_ids

