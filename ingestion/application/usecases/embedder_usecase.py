from typing import List, Dict
from ingestion.application.services.interfaces.embedding_interface import EmbeddingClient


class EmbedChunksUseCase:
    """
    Orchestrates embedding a list of chunk‐strings + metadata dict into
    a vector store via the injected EmbeddingClient.
    """

    def __init__(self, embedding_client: EmbeddingClient):
        self._emb_client = embedding_client

    def execute(
        self,
        chunks: List[str],
        metadata: Dict[str, any]
    ) -> List[str]:
        """
        For each chunk, pair it with the same metadata dict (or you can copy/extend metadata differently).
        Returns the list of generated IDs.
        """
        # If you want to attach the same metadata to each chunk, replicate it:
        metadatas = [metadata for _ in chunks]
        return self._emb_client.embed_texts(texts=chunks, metadatas=metadatas)
