import os

from crawler.application.services.image_upload_service import ImageUploadService
from crawler.domain.entities.page_image import PageImageEntity
from crawler.infrastructure.repositories.page_image_repository import PageImageRepository
from crawler.infrastructure.repositories.page_repository import PageRepository


class ImageUploadUseCase:
    def __init__(self, upload_service: ImageUploadService, image_repository: PageImageRepository,
                 page_repository: PageRepository):
        self.upload_service = upload_service
        self.image_repository = image_repository
        self.page_repository = page_repository

    def execute(self, page_image_entity: PageImageEntity) -> dict:
        # 1) Use the repository‐provided 'local_path' (absolute on‐disk path):
        local_path = page_image_entity.local_path
        if not local_path or not os.path.exists(local_path):
            msg = f"File not found for image ID {page_image_entity.id}: {local_path}"
            self.image_repository.mark_error(page_image_entity.id, msg)
            return {
                "image_id": page_image_entity.id,
                "status": "error",
                "error": msg
            }

        # 2) Fetch related Page data from the entity:
        #    Since `page_image_entity.page` is just the page_id (int),
        #    you may need to have a synchronous repo call to get the PageEntity
        #    if you need title or entity_id. For example:
        page_entity = self.page_repository.get_by_id(page_image_entity.page)
        title = page_entity.title or ""
        entity_id = page_entity.entity_id

        try:
            api_response = self.upload_service.upload_desktop_file(
                title=title,
                image_path=local_path,
                entity_id=entity_id
            )
            self.image_repository.mark_uploaded(page_image_entity.id, api_response)
            return {
                "image_id": page_image_entity.id,
                "status": "success",
                "api_response": api_response
            }
        except Exception as e:
            self.image_repository.mark_error(page_image_entity.id, str(e))
            return {
                "image_id": page_image_entity.id,
                "status": "error",
                "error": str(e)
            }