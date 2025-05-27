from abc import ABC, abstractmethod
from typing import List, Optional
from crawler.domain.entities.page import PageEntity
from crawler.domain.entities.page_image import PageImageEntity


class PageImageRepositoryInterface(ABC):
    @abstractmethod
    def create(self, page: PageImageEntity) -> PageImageEntity:
        """ creates a new page image object """
        pass

    @abstractmethod
    async def create_async(self, page: PageImageEntity) -> PageImageEntity:
        """ asynchronously creates page image objects """
        pass

    @abstractmethod
    def filter(self, **data) -> List[PageImageEntity]:
        """ returns a filtered queryset of page image objects """
        pass

    @abstractmethod
    def get_by_id(self, page_image_id: int) -> Optional[PageImageEntity]:
        """ returns a page image object by id """
        pass

    @abstractmethod
    async def get_by_id_async(self, page_image_id: int) -> Optional[PageImageEntity]:
        """ asynchronously gets a page image objects """
        pass

    @abstractmethod
    async def aexists(self, **lookup) -> bool:
        """ asynchronously checks if a page image exists by given parameters """
        pass