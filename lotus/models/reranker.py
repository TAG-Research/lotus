from abc import ABC, abstractmethod


class Reranker(ABC):
    """Abstract class for reranker models."""

    def _init__(self):
        pass

    @abstractmethod
    def __call__(self, query: str, docs: list[str], k: int) -> list[int]:
        """Invoke the reranker.

        Args:
            query (str): The query to use for reranking.
            docs (list[str]): A list of documents to rerank.
            k (int): The number of documents to keep after reranking.

        Returns:
            list[int]: The indicies of the reranked documents.
        """
        pass
