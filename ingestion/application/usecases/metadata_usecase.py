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


class MetadataUseCase:
    """
    Builds structured metadata (Excel + DB title) for a scraped Persian-language car page.
    Fixes:
      • “Value” column is explicitly read/​cast to object dtype so the dash “-” can be written
        without the FutureWarning that pandas ≥2.1 emits.
      • Fallback for missing values is consistently "-" (same symbol you instruct the LLM to return).
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
        # self._executor = ThreadPoolExecutor(max_workers=1)

    async def execute(self, page_id: int, content: str):
        page = await self.repository.get_by_id_async(page_id)
        if not page:
            raise ValueError(f"Page with ID {page_id} not found.")

        return await self.build_metadata(page, content)

    async def build_metadata(self, page_obj, content):

        if page_obj.title:
            car_model = page_obj.title.strip()
        else:
            if not content:
                raise ValueError("No content provided for metadata generation.")

            # --- Call the LLM -------------------------------------------------
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

            # Read the Excel template with a string-capable “Value” column
            df_template = pd.read_excel(
                self.dataframe_path,
                dtype={"Value": "object"}      # <-- FIX: makes “Value” accept strings like “-”
            )

            # Update DB & run downstream services
            page_obj.title = car_model
            await self.repository.update_async(page_obj)

            # Map Persian labels to JSON keys
            field_mapping = {
                "محور محرک": "محور_محرک",
                "گیربکس تک سرعته کاهشی": "گیربکس_تک_سرعته_کاهشی",
                "توان موتور": "توان_موتور",
                "سرعت": "سرعت",
                "شتاب": "شتاب",
                "ظرفیت باتری": "ظرفیت_باتری",
                "حداکثر مسافت پیمایش با شارژ کامل": "حداکثر_مسافت_پیمایش_با_شارژ_کامل",
                "مدت زمان شارژ عادی DC": "مدت_زمان_شارژ_عادی_DC",
                "مدت زمان شارژ سریع AC": "مدت_زمان_شارژ_سریع_AC",
                "گشتاور": "گشتاور",
                "طول (میلیمتر)": "طول_میلیمتر",
                "عرض(میلیمتر)": "عرض_میلیمتر",
                "ارتفاع": "ارتفاع",
                "نوع شاسی": "نوع_شاسی",
                "سیستم تعلیق جلو": "سیستم_تعلیق_جلو",
                "سیستم تعلیق عقب": "سیستم_تعلیق_عقب",
                "فرمان": "فرمان",
                "سیستم کروز کنترل": "سیستم_کروز_کنترل",
                "ایربگ راننده": "ایربگ_راننده",
                "ایربگ سرنشین جلو": "ایربگ_سرنشین_جلو",
                "ترمز ABS": "ترمز_ABS",
                "ترمز EBD": "ترمز_EBD",
                "کنترل پایداری ESP": "کنترل_پایداری_ESP",
                "ترمز جلو": "ترمز_جلو",
                "ترمز عقب": "ترمز_عقب",
                "استارت": "استارت",
                "صفحه نمایش مرکزی": "صفحه_نمایش_مرکزی",
                "دوربین عقب": "دوربین_عقب",
                "مه شکن عقب": "مه_شکن_عقب",
                "سنسور پارک جلو": "سنسور_پارک_جلو",
                "سنسور پارک عقب": "سنسور_پارک_عقب",
                "سنسور باران": "سنسور_باران",
                "تعداد بلندگو": "تعداد_بلندگو",
                "صندلی راننده": "صندلی_راننده",
                "تعداد صندلی": "تعداد_صندلی",
                "تهویه خودکار": "تهویه_خودکار",
                "GPS": "GPS",
                "بلوتوث": "بلوتوث",
                "USB": "USB",
                "ویژگی ها": "ویژگی_ها",
                "کلاس بدنه خودرو": "کلاس_بدنه_خودرو",
                "نوع و کاربری": "نوع_و_کاربری",
                "سیستم": "سیستم",
                "تیپ": "تیپ",
                "مدل": "مدل",
                "نوع سوخت": "نوع_سوخت",
                "تعداد سیلندر": "تعداد_سیلندر",
                "تعداد محور": "تعداد_محور",
                "تعداد چرخ": "تعداد_چرخ",
                "ظرفیت": "ظرفیت",
                "حجم سیلندر": "حجم_سیلندر",
                "کشور سازنده": "کشور_سازنده",
            }

            #Fill in the template
            for idx, row in df_template.iterrows():
                title = row["Title"]
                json_key = field_mapping.get(title)
                value = result.get(json_key, "-") if json_key else "-"
                df_template.at[idx, "Value"] = value

            # Save the filled-out sheet
            safe_name = car_model or page_obj.title or f"page_{page_obj.id}"
            output_file = os.path.join(
                settings.MEDIA_ROOT,
                f"car_specs_{safe_name}.xlsx"
            )
            df_template.to_excel(output_file, index=False)

            await self.id_matcher_service.find_best_match(page_obj.id)

            # (Optional asynchronous upload – left commented as in original)
            # loop = asyncio.get_running_loop()
            # try:
            #     upload_response = await loop.run_in_executor(
            #         self._executor,
            #         lambda: self.api_service.import_excel(
            #             path="api/services/app/CarsAndBrandService/Import",
            #             excel_path=output_file,
            #             product_id=page_obj.entity_id,
            #             type_value=2
            #         )
            #     )
            # except Exception as e:
            #     raise RuntimeError(f"Failed to upload Excel: {e}") from e

        #Return a lightweight summary
        return {
            "source": page_obj.source,
            "url": page_obj.url,
            "car_model": car_model,
            "page_id": page_obj.id,
        }
