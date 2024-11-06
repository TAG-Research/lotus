import os
import pickle
from abc import abstractmethod
from typing import Any

import faiss
import numpy as np
from numpy.typing import NDArray

from lotus.models.rm import RM
from lotus.types import RMOutput


class FaissRM(RM):
    def __init__(self, factory_string: str = "Flat", metric=faiss.METRIC_INNER_PRODUCT):
        super().__init__()
        self.factory_string = factory_string
        self.metric = metric
        self.index_dir: str | None = None
        self.faiss_index: faiss.Index | None = None
        self.vecs: NDArray[np.float64] | None = None

    def index(self, docs: list[str], index_dir: str, **kwargs: dict[str, Any]) -> None:
        vecs = self._embed(docs)
        self.faiss_index = faiss.index_factory(vecs.shape[1], self.factory_string, self.metric)
        self.faiss_index.add(vecs)
        self.index_dir = index_dir

        os.makedirs(index_dir, exist_ok=True)
        with open(f"{index_dir}/vecs", "wb") as fp:
            pickle.dump(vecs, fp)
        faiss.write_index(self.faiss_index, f"{index_dir}/index")

    def load_index(self, index_dir: str) -> None:
        self.index_dir = index_dir
        self.faiss_index = faiss.read_index(f"{index_dir}/index")
        with open(f"{index_dir}/vecs", "rb") as fp:
            self.vecs = pickle.load(fp)

    def get_vectors_from_index(self, index_dir: str, ids: list[int]) -> NDArray[np.float64]:
        with open(f"{index_dir}/vecs", "rb") as fp:
            vecs: NDArray[np.float64] = pickle.load(fp)
        return vecs[ids]

    def __call__(self, queries: str | list[str] | NDArray[np.float64], K: int, **kwargs: dict[str, Any]) -> RMOutput:
        if isinstance(queries, str):
            queries = [queries]

        if isinstance(queries[0], str):
            embedded_queries = self._embed([str(q) for q in queries])
        else:
            embedded_queries = np.asarray(queries, dtype=np.float32)

        if self.faiss_index is None:
            raise ValueError("Index not loaded")

        distances, indices = self.faiss_index.search(embedded_queries, K)
        return RMOutput(distances=distances, indices=indices)

    @abstractmethod
    def _embed(self, docs: list[str]) -> NDArray[np.float64]:
        pass
