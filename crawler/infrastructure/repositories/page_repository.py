from typing import List, Optional

from asgiref.sync import sync_to_async
from django.core.files.base import ContentFile
from django.utils.text import slugify
from crawler.domain.entities.page import PageEntity
from crawler.domain.interface.page_repository_interface import PageRepositoryInterface
from crawler.infrastructure.models import Page as PageModel


class PageRepository(PageRepositoryInterface):

    def _to_entity(self, page_model: PageModel) -> PageEntity:
        return PageEntity(
            id=page_model.id,
            source=page_model.source,
            title=page_model.title,
            markdown_file=page_model.markdown_file.name
                            if page_model.markdown_file else None,
            url=page_model.url,
            created_at=page_model.created_at,
            updated_at=page_model.updated_at,
            deleted_at=page_model.deleted_at,
            images=[],  # populate via image repo if you need them
        )

    def create(self, page: PageEntity) -> PageEntity:
        # Instantiate a **new instance** of the Django model:
        page_instance = PageModel(
            source=page.source,
            title=page.title,
            url=page.url,
        )

        if page.markdown_file:
            filename = f"{slugify(page.url)}.md"
            if isinstance(page.markdown_file, ContentFile):
                page_instance.markdown_file = page.markdown_file
                page_instance.markdown_file.name = (
                    page.markdown_file.name or filename
                )
            else:
                page_instance.markdown_file.save(
                    filename,
                    ContentFile(page.markdown_file),
                )

        # Now this is an instance, so save() has a proper `self`:
        page_instance.save()

        # And convert that instance back to your domain entity:
        return self._to_entity(page_instance)

    async def create_async(self, page: PageEntity) -> PageEntity:
        """
        Run the sync create() method in a thread so the DB call
        happens outside the async event-loop.
        """
        return await sync_to_async(self.create, thread_sensitive=True)(page)

    def filter(self, **data) -> List[PageEntity]:
        return [self._to_entity(m) for m in PageModel.objects.filter(**data)]

    def get_by_id(self, page_id: int) -> Optional[PageEntity]:
        try:
            m = PageModel.objects.get(id=page_id)
        except PageModel.DoesNotExist:
            return None
        return self._to_entity(m)

    async def get_by_id_async(self, page_id: int) -> Optional[PageEntity]:
        return await sync_to_async(self.get_by_id, thread_sensitive=True)(page_id)

    async def aexists(self, **lookup) -> bool:
        return await sync_to_async(PageModel.objects.filter(**lookup).exists, thread_sensitive=True)()