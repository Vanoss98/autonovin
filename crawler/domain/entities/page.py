from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Union
from django.core.files.base import ContentFile


@dataclass
class PageEntity:
    id: Optional[int] = None
    entity_id: Optional[int] = None
    source: Optional[str] = None
    title: Optional[str] = None
    markdown_file: Optional[Union[str, ContentFile]] = None
    url: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None
    images: List["PageImageEntity"] = field(default_factory=list)