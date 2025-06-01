from autonovin.common.utils import link_remover
from langchain_core.prompts import PromptTemplate


class MetadataUseCase:
    def __init__(self, llm_service, repository):
        self.llm_service = llm_service
        self.repository = repository

    async def execute(self, page_id: int, content: str):
        # Retrieve the page by its ID
        page = await self.repository.get_by_id_async(page_id)
        if not page:
            raise ValueError(f"Page with ID {page_id} not found.")

        # Generate metadata
        metadata = await self.build_metadata(page, content)

        return metadata

    async def build_metadata(self, page_obj, content):
        if page_obj.title:  # Skip LLM if we already have a title
            car_model = page_obj.title.strip()
        else:
            if not content:  # Content should already be provided
                raise ValueError("No content provided for metadata generation.")

            content = link_remover(content)

            prompt_template = """
            You are an expert in automotive documents.
            Given the following document, extract the car model mentioned.
            Return only the car model name as a plain string.
            {docs}
            """
            prompt = PromptTemplate(template=prompt_template, input_variables=["docs"])
            formatted_prompt = prompt.format(docs=content)

            llm_response = self.llm_service.invoke_llm(formatted_prompt)
            car_model = llm_response.strip()
            page_obj.title = car_model
            page_obj = await self.repository.update_async(page_obj)

        return {
            "source": page_obj.source,
            "url": page_obj.url,
            "car_model": car_model,
            "page_id": page_obj.id,
        }
