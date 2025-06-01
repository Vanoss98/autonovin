from abc import ABC, abstractmethod
from typing import List, Optional
from crawler.domain.entities.page import PageEntity


class PageRepositoryInterface(ABC):
    @abstractmethod
    def create(self, page: PageEntity) -> PageEntity:
        """ creates a new page object """
        pass

    @abstractmethod
    async def create_async(self, page: PageEntity) -> PageEntity:
        """ asynchronously creates page objects """
        pass

    @abstractmethod
    def update(self, page_entity: PageEntity) -> PageEntity:
        """
        Update an existing PageModel in the database using fields from the PageEntity.
        Returns a fresh PageEntity based on the saved model.
        """
        pass

    @abstractmethod
    async def update_async(self, page_entity: PageEntity) -> PageEntity:
        pass

    @abstractmethod
    def filter(self, **data) -> List[PageEntity]:
        """ returns a filtered queryset of page objects """
        pass

    @abstractmethod
    def get_by_id(self, page_id: int) -> Optional[PageEntity]:
        """ returns a page object by id """
        pass

    @abstractmethod
    async def get_by_id_async(self, page_id: int) -> Optional[PageEntity]:
        """ asynchronously gets a page objects """
        pass