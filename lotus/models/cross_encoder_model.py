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
        batch_size: int = 32,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device: str = device
        self.batch_size: int = batch_size
        self.model = CrossEncoder(model, device=device)

    def __call__(self, query: str, docs: list[str], k: int) -> list[int]:
        results = self.model.rank(query, docs, top_k=k, batch_size=self.batch_size)
        return [int(result["corpus_id"]) for result in results]
