from celery import shared_task, chain
from asgiref.sync import async_to_sync
from ingestion.application.services.chunker_service import ChunkerService
from ingestion.application.services.openai_service import OpenaiService
from ingestion.application.usecases.chunker_usecase import PageChunkingUseCase
from ingestion.application.usecases.metadata_usecase import MetadataUseCase
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

chunk_usecase = PageChunkingUseCase(chunker_service=ChunkerService(), repository=page_repo)
metadata_usecase = MetadataUseCase(llm_service=OpenaiService(), repository=page_repo)
# Wrap the PageCrawlerUseCase into a Celery task


@shared_task(bind=True)
def page_crawler_task(self, config):
    page_crawler_use_case = PageCrawlerUseCase(crawler=crawler_service, repository=page_repo,
                                               image_downloader=image_downloader, image_repository=image_repo)
    # Convert the async method to sync using async_to_sync
    page_ids = async_to_sync(page_crawler_use_case.execute)(config)
    for page_id in page_ids:
        page = async_to_sync(page_repo.get_by_id_async)(page_id)
        content = chunk_usecase.chunker_service.get_markdown_content(page)
        chunks = async_to_sync(chunk_usecase.execute)(page_id)
        metadata = async_to_sync(metadata_usecase.execute)(page_id, content)
        return {'chunks': chunks, 'metadata': metadata}


