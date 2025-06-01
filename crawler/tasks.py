from celery import shared_task, chain
from asgiref.sync import async_to_sync
from ingestion.application.services.chunker_service import ChunkerService
from ingestion.application.services.openai_service import OpenaiService
from ingestion.application.usecases.chunker_usecase import PageChunkingUseCase
from ingestion.application.usecases.metadata_usecase import MetadataUseCase
from .application.services.crawl4ai_crawler_service import Crawl4aiCrawlerService
from .application.services.entity_id_matcher_service import EntityIdMatcherService
from .application.services.image_download_service import ImageDownloadService
from .application.services.image_upload_service import ImageUploadService
from .application.use_cases.image_upload_usecase import ImageUploadUseCase
from .application.use_cases.page_crawler import PageCrawlerUseCase
from .infrastructure.repositories.page_repository import PageRepository
from .infrastructure.repositories.page_image_repository import PageImageRepository

# Initialize repositories and services
page_repo = PageRepository()
image_repo = PageImageRepository()
crawler_service = Crawl4aiCrawlerService()
image_downloader = ImageDownloadService()
entity_id_matcher_service = EntityIdMatcherService(file_path="cars.xlsx", threshold=0.9, repository=page_repo)
chunk_usecase = PageChunkingUseCase(chunker_service=ChunkerService(), repository=page_repo)
metadata_usecase = MetadataUseCase(llm_service=OpenaiService(), repository=page_repo)
# Wrap the PageCrawlerUseCase into a Celery task


# Configure once at import time:
API_UPLOAD_URL = "https://order.autonovin.ir/api/services/app/AttachmentDetailService/Add"
AUTH_TOKEN = None

upload_service = ImageUploadService(api_url=API_UPLOAD_URL, auth_token=AUTH_TOKEN)
image_upload_usecase = ImageUploadUseCase(
    upload_service=upload_service,
    image_repository=image_repo,
    page_repository=page_repo
)


@shared_task(bind=True)
def upload_image_task(self, page_image_id: int):
    """
    Celery task that:
      1) Loads PageImageEntity (with .local_path filled in)
      2) Calls the use case, which streams that file to the API
      3) Returns a dict with status or error
    """
    page_image_entity = image_repo.get_by_id(page_image_id)
    if not page_image_entity:
        return {"image_id": page_image_id, "status": "error", "error": "PageImage not found"}

    return image_upload_usecase.execute(page_image_entity)


@shared_task(bind=True)
def page_crawler_task(self, config):
    page_crawler_use_case = PageCrawlerUseCase(
        crawler=crawler_service,
        repository=page_repo,
        image_downloader=image_downloader,
        image_repository=image_repo
    )
    page_ids = async_to_sync(page_crawler_use_case.execute)(config)

    overall = []
    for page_id in page_ids:
        page = async_to_sync(page_repo.get_by_id_async)(page_id)
        content = chunk_usecase.chunker_service.get_markdown_content(page)
        chunks = async_to_sync(chunk_usecase.execute)(page_id)
        metadata = async_to_sync(metadata_usecase.execute)(page_id, content)
        async_to_sync(entity_id_matcher_service.find_best_match)(page_id)
        images = async_to_sync(image_repo.list_by_page_id_async)(page_id)
        for img in images:
            upload_image_task.delay(img.id)

        overall.append({
            "page_id": page_id,
            "chunks": chunks,
            "metadata": metadata,
            "images_enqueued_for_upload": [img.id for img in images],
        })

    return {"results": overall}


