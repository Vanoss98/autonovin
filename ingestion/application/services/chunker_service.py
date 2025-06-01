from django.core.files.base import ContentFile

from autonovin.common.utils import link_remover
from langchain_text_splitters import MarkdownHeaderTextSplitter

from crawler.domain.entities.page import PageEntity


class ChunkerService:

    def chunk(self, page_obj):
        # Ensure we're working with a PageEntity
        if not isinstance(page_obj, PageEntity):
            raise ValueError("Expected a PageEntity object, but got a different type.")

        if not page_obj.markdown_file:
            raise ValueError("No markdown_file attached to this Page object.")

        # Get the content of the markdown file
        content = link_remover(self.get_markdown_content(page_obj))

        # Perform chunking based on headers
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        docs = splitter.split_text(content)

        return [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]

    def get_markdown_content(self, page: PageEntity):
        if isinstance(page.markdown_file, str):
            # It's a file path (e.g., 'markdowns/some_file.md')
            try:
                with open(page.markdown_file, 'r', encoding='utf-8') as file:
                    return file.read()
            except FileNotFoundError:
                return f"File not found at {page.markdown_file}"
            except IOError:
                return f"Error reading file at {page.markdown_file}"
        elif isinstance(page.markdown_file, ContentFile):
            # It's a ContentFile object, so we can directly read its content
            return page.markdown_file.read().decode('utf-8')
        else:
            return "No markdown file attached"
