from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import URLPatternFilter, FilterChain
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


class Crawl4aiCrawlerService:
    def __init__(self):
        pass

    async def execute(self, config: dict):
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
        return results








