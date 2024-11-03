import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from lotus.models.faiss_rm import FaissRM


class SentenceTransformersRM(FaissRM):
    def __init__(self, model: str = "intfloat/e5-base-v2"):
        super().__init__()
        self.model: str = model
        self.transformer: SentenceTransformer = SentenceTransformer(model)

    def _embed(self, docs: list[str]) -> NDArray[np.float64]:
        return self.transformer.encode(docs, convert_to_tensor=True).cpu().numpy()


if __name__ == "__main__":
    rm = SentenceTransformersRM()
    docs = ["Machine Learning", "Quantum Physics"]
    index_dir = "index_dir"
    query = "Quantum Mechanics"
    rm.index(docs, index_dir)
    print(rm(query, 2))

    queries = ["Artifical Intelligence", "Quantum Mechanics"]
    print(rm(queries, 2))
