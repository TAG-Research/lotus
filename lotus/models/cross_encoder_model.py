import torch
from sentence_transformers import CrossEncoder

from lotus.models.reranker import Reranker


class CrossEncoderModel(Reranker):
    """CrossEncoder reranker model.

    Args:
        model (str): The name of the reranker model to use.
        device (str): What device to keep the model on.
    """

    def __init__(
        self,
        model: str = "mixedbread-ai/mxbai-rerank-large-v1",
        device: str | None = None,
        **kwargs,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = CrossEncoder(model, device=device, **kwargs)

    def __call__(self, query: str, docs: list[str], k: int) -> list[int]:
        results = self.model.rank(query, docs, top_k=k)
        results = [result["corpus_id"] for result in results]
        return results
