import asyncio
import os
from django.conf import settings
from autonovin.common.utils import link_remover
import pandas as pd
from crawler.application.services.api_service import ApiService
from crawler.application.services.entity_id_matcher_service import EntityIdMatcherService
from crawler.domain.interface.page_repository_interface import PageRepositoryInterface
from ingestion.domain.entities.car_spec import CarSpecs
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class MetadataUseCase:
    """
    Builds structured metadata (Excel + DB title) for a scraped Persian-language car page.
    Fixes:
      • “Value” column is explicitly read/cast to object dtype so the dash “-” can be written
        without the FutureWarning that pandas ≥2.1 emits.
      • Fallback for missing values is consistently "-" (same symbol you instruct the LLM to return).
      • Excel files are saved under MEDIA_ROOT/car_specs/, creating the folder if needed.
    """

    def __init__(
        self,
        llm_service,
        repository: "PageRepositoryInterface",
        dataframe_path: str,
        api_service: "ApiService",
        id_matcher_service: "EntityIdMatcherService",
    ):
        self.llm_service = llm_service
        self.repository = repository
        self.dataframe_path = os.path.join(settings.MEDIA_ROOT, dataframe_path)
        self.api_service = api_service
        self.id_matcher_service = id_matcher_service

    async def execute(self, page_id: int, content: str):
        page = await self.repository.get_by_id_async(page_id)
        if not page:
            raise ValueError(f"Page with ID {page_id} not found.")
        return await self.build_metadata(page, content)

    async def build_metadata(self, page_obj, content):
        # Determine car_model from existing title or via LLM
        if page_obj.title:
            car_model = page_obj.title.strip()
        else:
            if not content:
                raise ValueError("No content provided for metadata generation.")

            content = link_remover(content)
            prompt_template = """
            You are an expert in automotive documents.
            You are given an unstructured Persian document about a car including model, specifications, dimensions,
            performance, and features.
            Extract the relevant information and return it as a JSON object with the following fields as keys:

            {format_instructions}

            If a required field is missing, put a single dash "-" as the value.
            Document:
            {docs}
            """

            result = self.llm_service.invoke_with_json_output(
                template=prompt_template,
                output_structure=CarSpecs,
                doc_text=content,
            )
            car_model = result.get("car_title", "").strip()

            # Load Excel template, ensuring "Value" can hold "-"
            df_template = pd.read_excel(
                self.dataframe_path,
                dtype={"Value": "object"}
            )

            # Update DB title and persist
            page_obj.title = car_model
            await self.repository.update_async(page_obj)

            # Map Persian labels → JSON keys
            field_mapping = {
                # ... (same mapping as before) ...
            }

            # Fill in the template
            for idx, row in df_template.iterrows():
                title = row["Title"]
                json_key = field_mapping.get(title)
                df_template.at[idx, "Value"] = result.get(json_key, "-") if json_key else "-"

            # Determine safe filename
            safe_name = car_model or f"page_{page_obj.id}"

            # Ensure MEDIA_ROOT/car_specs exists
            output_dir = Path(settings.MEDIA_ROOT) / "car_specs"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Build full output path inside that folder
            output_file = output_dir / f"car_specs_{safe_name}.xlsx"

            # Save the filled-out sheet
            df_template.to_excel(output_file, index=False)

            # Run ID matcher
            await self.id_matcher_service.find_best_match(page_obj.id)

        # Return summary
        return {
            "source": page_obj.source,
            "url": page_obj.url,
            "car_model": car_model,
            "page_id": page_obj.id,
        }