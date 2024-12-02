import faiss
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from lotus.dtype_extensions import convert_to_base_data
from lotus.models.faiss_rm import FaissRM


class SentenceTransformersRM(FaissRM):
    def __init__(
        self,
        model: str = "intfloat/e5-base-v2",
        max_batch_size: int = 64,
        normalize_embeddings: bool = True,
        device: str | None = None,
        factory_string: str = "Flat",
        metric=faiss.METRIC_INNER_PRODUCT,
    ):
        super().__init__(factory_string, metric)
        self.model: str = model
        self.max_batch_size: int = max_batch_size
        self.normalize_embeddings: bool = normalize_embeddings
        self.transformer: SentenceTransformer = SentenceTransformer(model, device=device)

    def _embed(self, docs: pd.Series | list) -> NDArray[np.float64]:
        all_embeddings = []
        for i in range(0, len(docs), self.max_batch_size):
            batch = docs[i : i + self.max_batch_size]
            _batch = convert_to_base_data(batch)
            torch_embeddings = self.transformer.encode(
                _batch, convert_to_tensor=True, normalize_embeddings=self.normalize_embeddings
            )
            assert isinstance(torch_embeddings, torch.Tensor)
            cpu_embeddings = torch_embeddings.cpu().numpy()
            all_embeddings.append(cpu_embeddings)
        return np.vstack(all_embeddings)
