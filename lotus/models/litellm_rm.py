import faiss
import numpy as np
from litellm import embedding
from litellm.types.utils import EmbeddingResponse
from numpy.typing import NDArray

from lotus.models.faiss_rm import FaissRM


class LiteLLMRM(FaissRM):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        max_batch_size: int = 64,
        factory_string: str = "Flat",
        metric=faiss.METRIC_INNER_PRODUCT,
    ):
        super().__init__(factory_string, metric)
        self.model: str = model
        self.max_batch_size: int = max_batch_size

    def _embed(self, docs: list[str]) -> NDArray[np.float64]:
        all_embeddings = []
        for i in range(0, len(docs), self.max_batch_size):
            batch = docs[i : i + self.max_batch_size]
            response: EmbeddingResponse = embedding(model=self.model, input=batch)
            embeddings = np.array([d["embedding"] for d in response.data])
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)
