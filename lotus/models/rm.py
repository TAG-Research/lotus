from abc import ABC, abstractmethod
from typing import Any


class RM(ABC):
    """Abstract class for retriever models."""

    def _init__(self):
        pass

    @abstractmethod
    def index(self, docs: list[str], index_dir: str, **kwargs: dict[str, Any]) -> None:
        """Create index and store it to a directory.

        Args:
            docs (list[str]): A list of documents to index.
            index_dir (str): The directory to save the index in.
        """
        pass

    @abstractmethod
    def load_index(self, index_dir: str) -> None:
        """Load the index into memory.

        Args:
            index_dir (str): The directory of where the index is stored.
        """
        pass

    @abstractmethod
    def get_vectors_from_index(self, index_dir: str, ids: list[int]) -> list:
        """Get the vectors from the index.

        Args:
            index_dir (str): Directory of the index.
            ids (list[int]): The ids of the vectors to retrieve

        Returns:
            list: The vectors matching the specified ids.
        """

        pass

    @abstractmethod
    def __call__(
        self,
        queries: str | list[str] | list[list[float]],
        k: int,
        **kwargs: dict[str, Any],
    ) -> tuple[list[float], list[int]]:
        """Run top-k search on the index.

        Args:
            queries (str | list[str] | list[list[float]]): Either a query or a list of queries or a 2D FP32 array.
            k (int): The k to use for top-k search.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            tuple[list[float], list[int]]: A tuple of (distances, indices) of the top-k vectors
        """
        pass
