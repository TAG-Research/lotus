import numpy as np
from litellm import embedding
from numpy.typing import NDArray
from litellm.types.utils import EmbeddingResponse

from lotus.models.faiss_rm import FaissRM


class LiteLLMRM(FaissRM):
    def __init__(self, model: str = "text-embedding-3-small"):
        super().__init__()
        self.model: str = model

    def _embed(self, docs: list[str]) -> NDArray[np.float64]:
        response: EmbeddingResponse = embedding(model=self.model, input=docs)
        embeddings = np.array([d["embedding"] for d in response.data])
        return embeddings



if __name__ == "__main__":
    rm = LiteLLMRM()
    docs = ["Machine Learning", "Quantum Physics"]
    index_dir = "index_dir"
    query = "Quantum Mechanics"
    rm.index(docs, index_dir)
    print(rm(query, 2))

    queries = ["Artifical Intelligence", "Quantum Mechanics"]
    print(rm(queries, 2))
