from abc import ABC, abstractmethod
from typing import List, Dict


class EmbeddingClient(ABC):
    """
    A port/interface for anything that can take text chunks + metadata,
    embed them via some vector store, and return assigned IDs.
    """

    @abstractmethod
    def embed_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, any]],
    ) -> List[str]:
        """
        Given a list of text strings and a parallel list of metadata dicts,
        embed each pair and return a list of IDs under which they were stored.
        """
        raise NotImplementedError
