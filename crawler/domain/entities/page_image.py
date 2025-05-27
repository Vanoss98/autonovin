from dataclasses import dataclass
from typing import Optional, Union
from datetime import datetime
from django.core.files.base import ContentFile


@dataclass
class PageImageEntity:
    id: int | None
    page: int
    url: str
    file: Optional[Union[str, ContentFile]] = None
    downloaded_at: Optional[datetime] = None

