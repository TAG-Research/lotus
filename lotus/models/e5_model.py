import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from lotus.models.rm import RM


class E5Model(RM):
    """E5 retriever model"""

    def __init__(self, model: str = "intfloat/e5-base-v2", device: Optional[str] = None, **kwargs):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(self.device)
        self.faiss_index = None
        self.index_dir = None
        self.docs = None
        self.kwargs = {"normalize": True, "index_type": "Flat", **kwargs}
        self.batch_size = 100
        self.vecs = None

        import faiss

        self.faiss = faiss

    def average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Perform average pooling over the last hidden state.

        Args:
            last_hidden_states: Hidden states from the model's last layer
            attention_mask: Attention mask.

        Returns:
            Average pool over the last hidden state.
        """

        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embed(self, docs: List[str], **kwargs: Dict[str, Any]) -> np.ndarray:
        """Run the embedding model.

        Args:
            docs: A list of documents to embed.

        Returns:
            Embeddings of the documents.
        """

        kwargs = {**self.kwargs, **kwargs}

        batch_size = kwargs.get("batch_size", self.batch_size)
        
        # Calculating the embedding dimension
        total_docs = len(docs)
        first_batch = self.tokenizer(docs[:1], return_tensors="pt", padding=True, truncation=True)
        embed_dim = self.model(**first_batch).last_hidden_state.size(-1)

        # Pre-allocate a tensor for all embeddings
        embeddings = torch.empty((total_docs, embed_dim), device=self.device)
        # Processing batches
        with torch.inference_mode():  # Slightly faster than torch.no_grad() for inference
            for i, batch_start in enumerate(tqdm(range(0, total_docs, batch_size))):
                batch = docs[batch_start : batch_start + batch_size]
                batch_dict = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                outputs = self.model(**batch_dict)
                batch_embeddings = self.average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
                embeddings[batch_start : batch_start + batch_size] = batch_embeddings
        if kwargs["normalize"]:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.numpy(force=True)

    def index(self, docs: List[str], index_dir: str, **kwargs: Dict[str, Any]) -> None:
        # Make index directory
        os.makedirs(index_dir, exist_ok=True)

        # Get document embeddings
        kwargs = {**self.kwargs, **kwargs}
        embeddings = self.embed(docs, **kwargs)
        d = embeddings.shape[1]
        index = self.faiss.index_factory(d, kwargs["index_type"], self.faiss.METRIC_INNER_PRODUCT)
        index.add(embeddings)

        # Store index and documents
        self.faiss.write_index(index, f"{index_dir}/index")
        with open(f"{index_dir}/docs", "wb") as fp:
            pickle.dump(docs, fp)
        with open(f"{index_dir}/vecs", "wb") as fp:
            pickle.dump(embeddings, fp)
        self.faiss_index = index
        self.docs = docs
        self.index_dir = index_dir
        self.vecs = embeddings

    def load_index(self, index_dir: str) -> None:
        self.index_dir = index_dir
        self.faiss_index = self.faiss.read_index(f"{index_dir}/index")
        with open(f"{index_dir}/docs", "rb") as fp:
            self.docs = pickle.load(fp)
        with open(f"{index_dir}/vecs", "rb") as fp:
            self.vecs = pickle.load(fp)

    @classmethod
    def get_vectors_from_index(self, index_dir: str, ids: List[int]) -> List:
        with open(f"{index_dir}/vecs", "rb") as fp:
            vecs = pickle.load(fp)

        return vecs[ids]

    def load_vecs(self, index_dir: str, ids: List[int]) -> List:
        """loads vectors to the rm and returns them
        Args:
            index_dir (str): Directory of the index.
            ids (List[int]): The ids of the vectors to retrieve

        Returns:
            The vectors matching the specified ids.
        """

        if self.vecs is None:
            with open(f"{index_dir}/vecs", "rb") as fp:
                self.vecs = pickle.load(fp)

        return self.vecs[ids]

    def __call__(
        self,
        queries: Union[str, List[str], List[List[float]]],
        k: int,
        **kwargs: Dict[str, Any],
    ) -> Tuple[List[float], List[int]]:
        if isinstance(queries, str):
            queries = [queries]

        if isinstance(queries[0], str):
            embedded_queries = self.embed(queries, **kwargs)
        else:
            embedded_queries = queries

        distances, indicies = self.faiss_index.search(embedded_queries, k)

        return distances, indicies
