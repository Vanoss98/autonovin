from celery import shared_task, chain
from asgiref.sync import async_to_sync
from autonovin.common.infrastructure.request_client import RequestsHttpClient
from ingestion.application.services.chroma_service import ChromaEmbeddingClient
from ingestion.application.services.chunker_service import ChunkerService
from ingestion.application.services.openai_service import OpenaiService
from ingestion.application.usecases.chunker_usecase import PageChunkingUseCase
from ingestion.application.usecases.embedder_usecase import EmbedChunksUseCase
from ingestion.application.usecases.metadata_usecase import MetadataUseCase
from .application.services.api_service import ApiService
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
metadata_usecase = MetadataUseCase(llm_service=OpenaiService(), repository=page_repo, dataframe_path="car_specs.xlsx",
                                   api_service=ApiService(http_client=RequestsHttpClient(),
                                                          base_url="https://isorder.iranfab.com"),
                                   id_matcher_service=entity_id_matcher_service)


chroma_client = ChromaEmbeddingClient(
    chroma_host="chroma",
    chroma_port=8000,
    collection_name="car_spec",
    embedding_model="text-embedding-3-large",
    embedding_dimensions=1024
)
embed_usecase = EmbedChunksUseCase(embedding_client=chroma_client)



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

def make_embed_usecase() -> EmbedChunksUseCase:
    """
    Build a fresh ChromaEmbeddingClient (+ EmbedChunksUseCase) every call.
    Called inside each task run so the Chroma collection UUID is always valid.
    """
    chroma_client = ChromaEmbeddingClient(
        chroma_host="chroma",            # service name from docker-compose
        chroma_port=8000,
        collection_name="car_spec",
        embedding_model="text-embedding-3-large",
        # embedding_dimensions=1024,     # let Chroma infer → fewer mismatch bugs
    )
    return EmbedChunksUseCase(embedding_client=chroma_client)


@shared_task(bind=True)
def page_crawler_task(self, config):
    embed_usecase = make_embed_usecase()
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
        # images = async_to_sync(image_repo.list_by_page_id_async)(page_id)
        # for img in images:
        #     upload_image_task.delay(img.id)
        try:
            chunk_ids = embed_usecase.execute(chunks, metadata)
        except Exception as e:
            # If embedding fails, you can choose to log or raise:
            raise RuntimeError(f"Failed embedding chunks for page {page_id}: {e}") from e

        overall.append({
            "page_id": page_id,
            "chunks": chunks,
            "metadata": metadata,
            # "images_enqueued_for_upload": [img.id for img in images],
        })

    return {"results": overall}


