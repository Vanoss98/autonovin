from django.utils.text import slugify
from django.core.files.base import ContentFile
from crawler.application.services.crawl4ai_crawler_service import Crawl4aiCrawlerService
from crawler.domain.entities.page import PageEntity
from crawler.infrastructure.repositories.page_repository import PageRepository


class PageCrawlerUseCase:
    def __init__(self, crawler, repository, image_downloader, image_repository):
        self.crawler = crawler
        self.repository = repository
        self.image_downloader = image_downloader
        self.image_repository = image_repository

    async def execute(self, config):
        results = await self.crawler.execute(config)
        # persist each successful page
        page_ids = []
        for res in results:
            if not res.success or not res.markdown:
                continue

            raw_md = getattr(res.markdown, "raw_markdown", str(res.markdown))
            title = getattr(getattr(res, "metadata", None), "title", None) or None

            # Wrap into a ContentFile (if your use-case signature expects it)
            filename = f"{slugify(res.url)}.md"
            md_file = ContentFile(raw_md, name=filename)

            # 3) Delegate to your async create use-case
            entity = PageEntity(
                id=None,
                source=config["start_url"],
                title=title,
                markdown_file=md_file,
                url=res.url)
            page = await self.repository.create_async(entity)
            page_ids.append(page.id)

            # Download images after creating the page
            await self.image_downloader.run(raw_md, page, self.image_repository)

        return page_ids




