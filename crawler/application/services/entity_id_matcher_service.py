import os
import re
import unicodedata
import pandas as pd
from difflib import SequenceMatcher
from django.conf import settings
from crawler.domain.interface.page_repository_interface import PageRepositoryInterface


class EntityIdMatcherService:
    def __init__(self, file_path: str, threshold: float, repository: PageRepositoryInterface):
        self.repository = repository
        self.excel_path = os.path.join(settings.MEDIA_ROOT, file_path)
        self.threshold = threshold

        # Verify the file exists
        if not os.path.isfile(self.excel_path):
            raise FileNotFoundError(f"Excel file not found: {self.excel_path}")

    @staticmethod
    def normalize_mixed_text(text: str) -> str:
        """
        1. Apply Unicode NFKC normalization.
        2. Insert spaces between Farsi (Arabic‐script) chars and Latin/digits.
        3. Collapse multiple spaces and strip edges.
        4. Lowercase.
        """
        text = unicodedata.normalize("NFKC", text)
        # Insert space between Farsi char and Latin/digit (both directions)
        text = re.sub(r'([\u0600-\u06FF])([A-Za-z0-9])', r'\1 \2', text)
        text = re.sub(r'([A-Za-z0-9])([\u0600-\u06FF])', r'\1 \2', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

    @classmethod
    def normalized_similarity(cls, a: str, b: str) -> float:
        """
        Compute a similarity score (0–1) between two strings after normalization.
        """
        na = cls.normalize_mixed_text(a)
        nb = cls.normalize_mixed_text(b)
        return SequenceMatcher(None, na, nb).ratio()

    async def find_best_match(self, page_id: int):
        """
        Reads the Excel at self.excel_path (expects columns 'Id' and 'Title').
        Compares `page_title` against each row’s Title (after normalization).
        Returns (best_entity_id, best_title, best_score) if best_score >= threshold;
        otherwise returns (None, None, 0.0).
        """
        # Read the Excel file
        df = pd.read_excel(self.excel_path)
        page = await self.repository.get_by_id_async(page_id)
        # Validate required columns
        if 'Id' not in df.columns or 'Title' not in df.columns:
            raise ValueError("Excel file must contain 'Id' and 'Title' columns.")

        best_score = 0.0
        best_entity = None
        best_title = None

        for _, row in df.iterrows():
            candidate_title = str(row['Title'])
            score = self.normalized_similarity(page.title, candidate_title)
            if score > best_score:
                best_score = score
                best_entity = row['Id']
                best_title = candidate_title

        if best_score >= self.threshold:
            page.entity_id = best_entity
            await self.repository.update_async(page)
        else:
            return "no match found"

