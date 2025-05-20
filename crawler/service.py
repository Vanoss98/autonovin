from asgiref.sync import sync_to_async
from django.utils.text import slugify

from .repository import PageRepository, PageImageRepository
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import URLPatternFilter, FilterChain
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from .serializers import CrawlConfigSerializer
import re
import aiohttp
import asyncio
from pathlib import Path
from django.core.files.base import ContentFile


class CrawlService:
    def __init__(self, page_repo=None, image_repo=None):
        self.repository = page_repo or PageRepository()
        self.serializer = CrawlConfigSerializer

    async def crawl(self, config: dict):
        include_filter = URLPatternFilter(patterns=config["include_patterns"])
        exclude_filter = URLPatternFilter(patterns=config["exclude_patterns"], reverse=True)
        filter_chain = FilterChain([include_filter, exclude_filter])

        deep_crawl = BFSDeepCrawlStrategy(
            max_depth=config['max_depth'],
            max_pages=config['max_pages'],
            include_external=False,
            filter_chain=filter_chain,
        )

        md_generator = DefaultMarkdownGenerator(options={"citations": False, "body_width": 0})

        run_config = CrawlerRunConfig(
            deep_crawl_strategy=deep_crawl,
            scraping_strategy=LXMLWebScrapingStrategy(),
            markdown_generator=md_generator,
            verbose=True,
            excluded_selector=config['excluded_selector'],
            excluded_tags=config['excluded_tags'],
        )

        # -------- run crawler --------------------------------------------
        async with AsyncWebCrawler() as crawler:
            results = await crawler.arun(url=config['start_url'], config=run_config)

        # -------- persist results ----------------------------------------
        for res in results:
            if not res.success or not res.markdown:
                continue

            raw_md = getattr(res.markdown, "raw_markdown", str(res.markdown))
            title = getattr(getattr(res, "metadata", None), "title", "")

            # --- build filename, wrap into ContentFile -------------------
            filename = f"{slugify(Path(res.url).as_posix())}.md"
            md_file = ContentFile(raw_md, name=filename)

            await self.repository.acreate({
                "url": res.url,
                "source": config["start_url"],
                "title": title,
                "markdown_file": md_file,  # <── FileField data
            })


class ImageDownloadService:
    IMG_PATTERN = re.compile(
        r'!\[[^\]]*?\]\s*'
        r'\(\s*(https?://[^\s)]+\.(?:jpe?g|png|gif|bmp|webp)(?:\?[^\s)]*)?)\s*\)',
        re.IGNORECASE | re.UNICODE,
    )

    def __init__(self, page_repo=None, image_repo=None):
        self.page_repo = page_repo or PageRepository()
        self.image_repo = image_repo or PageImageRepository()

    # ---------- public ----------------------------------------------------

    async def run(self, page_id: int) -> int:
        page = await self.page_repo.aget_by_id(page_id)
        if not page or not page.markdown_file:
            return 0

        md_text = await self._read_markdown(page)
        urls = self._extract_urls(md_text)
        if not urls:
            return 0

        sem = asyncio.Semaphore(5)

        async def _worker(url):
            async with sem:
                if await self.image_repo.aexists(page_id=page.id, url=url):
                    return False
                content = await self._download(url)
                if content is None:
                    return False
                await self._save_image(page, url, content)
                return True

        results = await asyncio.gather(*(_worker(u) for u in urls))
        return sum(results)            # number of new images saved

    # ---------- helpers ---------------------------------------------------

    async def _read_markdown(self, page) -> str:
        """Read markdown_file safely from the async context."""
        def _sync_read():
            with page.markdown_file.open("r") as f:
                return f.read()
        return await sync_to_async(_sync_read, thread_sensitive=True)()

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

    async def _save_image(self, page, url, content: bytes):
        filename = Path(url).name.split("?")[0] or "image"
        await self.image_repo.acreate({
            "page": page,
            "url": url,
            "file": ContentFile(content, name=filename),
        })
