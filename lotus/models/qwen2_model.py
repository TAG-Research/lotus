from typing import Any, Union
import numpy as np
import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from lotus.models.faiss_rm import FaissRM
import torch.nn.functional as F
from PIL import Image
from numpy.typing import NDArray
from lotus.types import RMOutput
from lotus.utils import fetch_image
import faiss

class Qwen2Model(FaissRM):
    """Qwen2 retriever model with vision-language capabilities"""

    def __init__(
        self,
        model: str = "Qwen/Qwen2-VL-2B-Instruct",
        max_batch_size: int = 64,
        normalize_embeddings: bool = True,
        device: str | None = None,
        factory_string: str = "Flat",
        metric=faiss.METRIC_INNER_PRODUCT,
    ):
        super().__init__(factory_string, metric)
        self.model = model
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.max_batch_size: int = max_batch_size  
        self.normalize_embeddings: bool = normalize_embeddings
        self.transformer = Qwen2VLForConditionalGeneration.from_pretrained(self.model).to(self.device).eval()
        self.processor = Qwen2VLProcessor.from_pretrained(self.model)

        import faiss
        self.faiss = faiss

    def _process_input(self, input_data: list[Union[str, Image.Image]]) -> dict:
        """Process different types of input data for the model."""
        texts = []
        images = []
        for data in input_data:
            try:
                image = fetch_image(data)
                texts.append("<|image_pad|>")
                images.append(image)
            except Exception as e:
                print(e)
                texts.append(data)
        if len(images) == 0:
            images = None
        processed_input = self.processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True).to(self.device)
        cache_position = torch.arange(0, len(texts))
        return self.transformer.prepare_inputs_for_generation(**processed_input, cache_position=cache_position, use_cache=False)

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


    def _embed(self, docs: list[Union[str, Image.Image]]) ->  NDArray[np.float64]:
        """Run the embedding model.

        Args:
            docs: A list of documents to embed (can be text strings, images, or base64 image strings)

        Returns:
            A numpy array of embeddings
        """
        all_embeddings = []
        with torch.inference_mode():
            for batch_start in tqdm(range(0, len(docs), self.max_batch_size)):
                batch = docs[batch_start : batch_start + self.max_batch_size]
                processed_input = self._process_input(batch)
                outputs = self.transformer(**processed_input, return_dict=True, output_hidden_states=True)
                batch_embeddings = self.average_pool(outputs.hidden_states[-1], processed_input["attention_mask"])
                assert isinstance(batch_embeddings, torch.Tensor)
                cpu_embeddings = batch_embeddings.cpu()
                all_embeddings.append(cpu_embeddings)
        all_embeddings = torch.stack(all_embeddings)
        if self.normalize_embeddings:
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
        torch.cuda.empty_cache()
        return all_embeddings.numpy(force=True)

    def __call__(self, queries: str | Image.Image | list[str | Image.Image] | NDArray[np.float64], K: int, **kwargs: dict[str, Any]) -> RMOutput:
        """Run top-k search on the index."""
        if self.faiss_index is None:
            raise ValueError("Index not loaded. Call load_index first.")

        if isinstance(queries, (str, Image.Image)):
            queries = [queries]

        if isinstance(queries[0], (str, Image.Image)):
            embedded_queries = self._embed(queries, **kwargs)
        else:
            embedded_queries = np.array(queries, dtype=np.float32)

        distances, indices = self.faiss_index.search(embedded_queries, K)
        return RMOutput(distances=distances, indices=indices)
