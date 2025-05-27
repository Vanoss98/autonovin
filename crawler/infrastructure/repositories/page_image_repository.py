# crawler/infrastructure/repositories/page_image_repository.py
from typing import List, Optional
from asgiref.sync import sync_to_async
from django.core.files.base import ContentFile
from django.utils.text import slugify

from crawler.domain.entities.page_image import PageImageEntity
from crawler.domain.interface.page_image_repository_interface import (
    PageImageRepositoryInterface,
)
from crawler.infrastructure.models import PageImage as PageImageModel, Page as PageModel


class PageImageRepository(PageImageRepositoryInterface):
    """Django's implementation of the image repo interface."""

    # ---------- mapping helpers ----------
    @staticmethod
    def _to_entity(model: PageImageModel) -> PageImageEntity:
        return PageImageEntity(
            id=model.id,
            page=model.page.id,
            url=model.url,
            file=model.file.name if model.file else None,
            downloaded_at=model.downloaded_at,
        )

    # ---------- sync methods ----------
    def create(self, image: PageImageEntity) -> PageImageEntity:
        if not image.id and not isinstance(image.file, ContentFile):
            raise ValueError("For new images, 'file' must be a ContentFile")

        page_model = PageModel.objects.get(id=image.page)  # image must carry page_id
        model = PageImageModel(page=page_model, url=image.url)

        if isinstance(image.file, ContentFile):
            filename = image.file.name  # Use the filename directly (not slugified version)
            print(f"Saving file with filename: {filename}")  # Debugging print statement
            model.file.save(filename, image.file)  # Save file with the correct filename
        elif isinstance(image.file, str):
            # if it's a string path you could copy or assign; skipping for brevity
            model.file.name = image.file

        model.save()
        return self._to_entity(model)

    def filter(self, **data) -> List[PageImageEntity]:
        return [self._to_entity(m) for m in PageImageModel.objects.filter(**data)]

    def get_by_id(self, page_image_id: int) -> Optional[PageImageEntity]:
        try:
            return self._to_entity(PageImageModel.objects.get(id=page_image_id))
        except PageImageModel.DoesNotExist:
            return None

    # ---------- async wrappers ----------
    async def create_async(self, image: PageImageEntity) -> PageImageEntity:
        return await sync_to_async(self.create, thread_sensitive=True)(image)

    async def get_by_id_async(self, image_id: int) -> Optional[PageImageEntity]:
        return await sync_to_async(self.get_by_id, thread_sensitive=True)(image_id)

    async def filter_async(self, **data) -> List[PageImageEntity]:
        return await sync_to_async(self.filter, thread_sensitive=True)(**data)

    async def aexists(self, **lookup) -> bool:
        return await sync_to_async(PageImageModel.objects.filter(**lookup).exists, thread_sensitive=True)()
