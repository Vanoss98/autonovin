from django.utils.text import slugify
from django.core.files.base import ContentFile
from crawler.application.services.crawl4ai_crawler_service import Crawl4aiCrawlerService
from crawler.domain.entities.page import PageEntity
from crawler.infrastructure.repositories.page_repository import PageRepository
from urllib.parse import urlparse, urlunparse, unquote


def _normalize_no_frag_query(u: str) -> str:  # NEW
    p = urlparse(u)
    p = p._replace(fragment="", query="")
    path = p.path.rstrip("/") or "/"
    p = p._replace(path=path)
    return urlunparse((p.scheme.lower(), p.netloc.lower(), p.path, "", "", ""))


def _is_seed_url(u: str, seed: str) -> bool:  # NEW
    return _normalize_no_frag_query(u) == _normalize_no_frag_query(seed)


class PageCrawlerUseCase:
    def __init__(self, crawler, repository, image_downloader, image_repository):
        self.crawler = crawler
        self.repository = repository
        self.image_downloader = image_downloader
        self.image_repository = image_repository

    async def execute(self, config):
        results = await self.crawler.execute(config)
        page_ids = []

        for res in results:
            # ⏭️ Skip the seed/listing page itself
            if _is_seed_url(getattr(res, "url", ""), config["start_url"]):  # NEW
                # optional: print(f"⏭️  Skipping seed URL: {res.url}")
                continue

            if not getattr(res, "success", False) or not getattr(res, "markdown", None):
                continue

            raw_md = getattr(res.markdown, "raw_markdown", str(res.markdown))
            title = getattr(getattr(res, "metadata", None), "title", None) or None

            filename = f"{slugify(res.url)}.md"
            md_file = ContentFile(raw_md, name=filename)

            entity = PageEntity(
                id=None,
                source=config["start_url"],
                title=title,
                markdown_file=md_file,
                url=res.url
            )
            page = await self.repository.create_async(entity)
            page_ids.append(page.id)

            # Download images only for saved (i.e., non-seed) pages
            await self.image_downloader.run(raw_md, page, self.image_repository)

        return page_ids
