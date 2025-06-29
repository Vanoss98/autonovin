import uuid
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from ingestion.application.services.interfaces.embedding_interface import EmbeddingClient


class ChromaEmbeddingClient(EmbeddingClient):
    """
    Implementation of EmbeddingClient that uses Chroma HTTP + OpenAIEmbeddings.
    """

    def __init__(
        self,
        chroma_host: str = "chroma",
        chroma_port: int = 8000,
        collection_name: str = "car_spec",
        embedding_model: str = "text-embedding-3-large",
        embedding_dimensions: int = 1024
    ):
        # Create a single shared HttpClient to talk to Chroma
        self._client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port,
            settings=Settings(allow_reset=False, anonymized_telemetry=False)
        )
        self._collection_name = collection_name
        # Configure the OpenAIEmbeddings function
        self._embedding_fn = OpenAIEmbeddings(
            model=embedding_model,
            dimensions=embedding_dimensions
        )

        # Instantiate a Chroma vector store that uses the above client + embeddings
        self._store = Chroma(
            client=self._client,
            collection_name=self._collection_name,
            embedding_function=self._embedding_fn
        )

    def embed_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, any]],
    ) -> List[str]:
        """
        For each text+metadata pair, generate a random UUID, add to Chroma, return list of IDs.
        """
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have the same length")

        ids = []
        # We can add them all at once, but to ensure unique IDs per-chunk:
        for text, meta in zip(texts, metadatas):
            chunk_id = f"chunk_{uuid.uuid4().hex}"
            # add_texts can accept one‐element lists
            self._store.add_texts(
                texts=[text],
                metadatas=[meta],
                ids=[chunk_id]
            )
            ids.append(chunk_id)

        return ids
