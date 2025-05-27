import asyncio
import aiohttp
import re
from pathlib import Path
from django.core.files.base import ContentFile
import random
from crawler.domain.entities.page_image import PageImageEntity


class ImageDownloadService:
    IMG_PATTERN = re.compile(
        r'\(\s*(https?://[^\s)]+\.(?:jpe?g|png|gif|bmp|webp)(?:\?[^\s)]*)?)\s*\)',
        re.IGNORECASE | re.UNICODE,
    )

    async def run(self, md_text: str, page, image_repo):
        urls = self._extract_urls(md_text)
        if not urls:
            return 0

        sem = asyncio.Semaphore(5)

        async def _worker(url):
            async with sem:
                if await image_repo.aexists(page_id=page.id, url=url):
                    return False
                content = await self._download(url)
                if content is None:
                    return False

                # Generate a random 8-digit number for the filename
                random_number = random.randint(10000000, 99999999)  # Random 8-digit number
                extension = Path(url).suffix.lower()  # Get the file extension (e.g., .jpg, .png)

                # Create the filename using page.id and random number for uniqueness
                filename = f"image_{page.id}_{random_number}{extension}"

                await self._save_image(page, url, content, image_repo, filename)
                return True

        results = await asyncio.gather(*(_worker(u) for u in urls))
        return sum(results)  # Return the number of new images saved

    def _extract_urls(self, markdown: str) -> list[str]:
        return self.IMG_PATTERN.findall(markdown or "")

    async def _download(self, url: str) -> bytes | None:
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(url, timeout=10) as resp:
                    resp.raise_for_status()
                    return await resp.read()
        except Exception:
            return None

    async def _save_image(self, page, url, content: bytes, image_repo, filename: str):
        entity = PageImageEntity(
            id=None,
            page=page.id,
            url=url,
            file=ContentFile(content, name=filename),
        )
        await image_repo.create_async(entity)


