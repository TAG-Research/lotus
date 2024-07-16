from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union


class RM(ABC):
    """Abstract class for retriever models."""

    def _init__(self):
        pass

    @abstractmethod
    def index(self, docs: List[str], index_dir: str, **kwargs: Dict[str, Any]) -> None:
        """Create index and store it to a directory.

        Args:
            docs (List[str]): A list of documents to index.
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
    def get_vectors_from_index(self, index_dir: str, ids: List[int]) -> List:
        """Get the vectors from the index.

        Args:
            index_dir (str): Directory of the index.
            ids (List[int]): The ids of the vectors to retrieve

        Returns:
            List: The vectors matching the specified ids.
        """

        pass

    @abstractmethod
    def __call__(
        self,
        queries: Union[str, List[str], List[List[float]]],
        k: int,
        **kwargs: Dict[str, Any],
    ) -> Tuple[List[float], List[int]]:
        """Run top-k search on the index.

        Args:
            queries (Union[str, List[str]]): Either a query or a list of queries or a 2D FP32 array.
            k (int): The k to use for top-k search.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Returns:
            Tuple[List[float], List[int]]: A tuple of (distances, indices) of the top-k vectors
        """
        pass
