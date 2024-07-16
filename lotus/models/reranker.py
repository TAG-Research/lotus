from abc import ABC, abstractmethod
from typing import List


class Reranker(ABC):
    """Abstract class for reranker models."""

    def _init__(self):
        pass

    @abstractmethod
    def __call__(self, query: str, docs: List[str], k: int) -> List[int]:
        """Invoke the reranker.

        Args:
            query (str): The query to use for reranking.
            docs (List[str]): A list of documents to rerank.
            k (int): The number of documents to keep after reranking.

        Returns:
            List[int]: The indicies of the reranked documents.
        """
        pass
