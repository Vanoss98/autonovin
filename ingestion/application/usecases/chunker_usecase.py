

class PageChunkingUseCase:
    def __init__(self, chunker_service, repository):
        self.chunker_service = chunker_service
        self.repository = repository

    async def execute(self, page_id: int):
        # Retrieve the page by its ID
        page = await self.repository.get_by_id_async(page_id)
        if not page:
            raise ValueError(f"Page with ID {page_id} not found.")

        # Chunk the page content
        chunks = self.chunker_service.chunk(page)

        return chunks
