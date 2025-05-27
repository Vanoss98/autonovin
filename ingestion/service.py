from django.conf import settings
from langchain_core.messages import SystemMessage
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import uuid
import chromadb
from chromadb.config import Settings
import os
import re
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from langchain_core.caches import BaseCache
from langchain_core.callbacks import Callbacks

# Set cache
set_llm_cache(InMemoryCache())

# Import required types so Pydantic can resolve them
_ = BaseCache
_ = Callbacks

# Rebuild the model
ChatOpenAI.model_rebuild()


# 1) link_remover: strip out all markdown links & images
def link_remover(md_text: str) -> str:
    md_text = re.sub(r'!\[.*?\]\(.*?\)', '', md_text)
    md_text = re.sub(r'\[.*?\]\(.*?\)', '', md_text)
    return md_text

# 2) chunk_page_markdown: read, clean, then split into text chunks
def chunk_page_markdown(page_obj, config):
    if not page_obj.markdown_file:
        raise ValueError("No markdown_file attached to this Page object.")
    page_obj.markdown_file.seek(0)
    content = page_obj.markdown_file.read()
    if isinstance(content, bytes):
        content = content.decode('utf-8')
    content = link_remover(content)
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    docs = splitter.split_text(content)
    return [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]

# 3) gen_answer_prompt: build LLM prompt
def gen_answer_prompt(docs: str) -> str:
    return (
        "You are an expert in automotive documents. "
        "Given the following document, extract the car model mentioned. "
        "Return only the car model name as a plain string.\n\n"
        f"{docs}"
    )

# 4) build_metadata: reuse title if set, else clean & call LLM once
def build_metadata(page_obj):
    if page_obj.title:  # skip LLM if we already have a title
        car_model = page_obj.title.strip()
    else:
        if not page_obj.markdown_file:
            raise ValueError("No markdown_file attached to this Page object.")
        page_obj.markdown_file.seek(0)
        content = page_obj.markdown_file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        content = link_remover(content)
        prompt = gen_answer_prompt(content)
        llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0, api_key=settings.OPENAI_API_KEY)
        sys_msg = SystemMessage(content=prompt)
        resp = llm.invoke([sys_msg])
        car_model = resp.content.strip()
        page_obj.title = car_model
        page_obj.save()

    return {
        "source": page_obj.source,
        "url": page_obj.url,
        "car_model": car_model,
        "page_id": page_obj.id,
    }

# 5) set up a single Chroma HTTP client (no reset on startup)
chroma_client = chromadb.HttpClient(
    host="chroma",
    port=8000,
    settings=Settings(allow_reset=False, anonymized_telemetry=False),
)

# 6) helper to embed one chunk + metadata
def embed_chunk_with_metadata(
    chunk_text: str,
    metadata: dict,
    collection_name: str = "chroma-test",
    embedding_model: str = "text-embedding-3-large",
    embedding_dimensions: int = 1024,
) -> str:
    embeddings = OpenAIEmbeddings(model=embedding_model, dimensions=embedding_dimensions)
    store = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    chunk_id = f"chunk_{uuid.uuid4().hex}"
    store.add_texts(texts=[chunk_text], metadatas=[metadata], ids=[chunk_id])
    return chunk_id

# 7) index_pages: full pipeline
def index_pages(queryset, config=None):
    print("Starting indexing …")
    config = config or {}
    if not config:
        print("No config provided, using defaults.")

    page_chunks = {}
    for page in queryset:
        print(f"\nProcessing Page {page.id} …")
        chunks = chunk_page_markdown(page, config)
        print(f"  Split into {len(chunks)} chunks.")

        meta = build_metadata(page)
        print(f"  Metadata built: {meta}")

        ids = []
        for text in chunks:
            cid = embed_chunk_with_metadata(text, meta)
            ids.append(cid)
        print(f"  Added {len(ids)} chunks to Chroma.")

        page_chunks[page.id] = ids

    return page_chunks