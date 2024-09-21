import os
import pickle
import base64
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from lotus.models.rm import RM


class CLIPModelRetriever(RM):
    """CLIP retriever model with multimodal (text & image) embedding support"""

    def __init__(self, model: str = "openai/clip-vit-base-patch32", device: Optional[str] = None, **kwargs):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model)
        self.model = CLIPModel.from_pretrained(model).to(self.device)
        self.faiss_index = None
        self.index_dir = None
        self.docs = None
        self.kwargs = {"normalize": True, "index_type": "Flat", **kwargs}
        self.batch_size = 5000
        self.vecs = None

        import faiss

        self.faiss = faiss

    def embed_text(self, texts: List[str], **kwargs: Dict[str, Any]) -> np.ndarray:
        """Run the text embedding model."""

        kwargs = {**self.kwargs, **kwargs}

        batch_size = kwargs.get("batch_size", self.batch_size)
        embeddings = []
        for i, batch_start in enumerate(tqdm(range(0, len(texts), batch_size))):
            batch = texts[batch_start : batch_start + batch_size]

            with torch.no_grad():
                inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = self.model.get_text_features(**inputs)
                embeddings.append(outputs)

        embeddings = torch.cat(embeddings, dim=0)
        if kwargs["normalize"]:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def embed_images(self, images: List[Any], **kwargs: Dict[str, Any]) -> np.ndarray:
        """Run the image embedding model."""

        kwargs = {**self.kwargs, **kwargs}

        batch_size = kwargs.get("batch_size", self.batch_size)
        embeddings = []
        for i, batch_start in enumerate(tqdm(range(0, len(images), batch_size))):
            batch = images[batch_start : batch_start + batch_size]

            with torch.no_grad():
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.model.get_image_features(**inputs)
                embeddings.append(outputs)

        embeddings = torch.cat(embeddings, dim=0)
        if kwargs["normalize"]:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def index(self, docs: List[str], index_dir: str, **kwargs: Dict[str, Any]) -> None:
        # Make index directory
        os.makedirs(index_dir, exist_ok=True)

        # Initialize embeddings storage
        kwargs = {**self.kwargs, **kwargs}
        d = None
        index = None

        # Process documents in batches
        for batch_start in tqdm(range(0, len(docs), self.batch_size)):
            batch_docs = docs[batch_start:batch_start + self.batch_size]
            
            # Separate text and image documents
            text_docs = []
            image_docs = []
            for doc in batch_docs:
                if doc.startswith("data:image"):
                    image_docs.append(self.base64_to_image(doc))
                else:
                    text_docs.append(doc)
            
            # Embed text and images separately
            text_embeddings = self.embed_text(text_docs, **kwargs) if text_docs else np.array([])
            image_embeddings = self.embed_images(image_docs, **kwargs) if image_docs else np.array([])
            
            # Combine embeddings
            if len(text_embeddings) > 0 and len(image_embeddings) > 0:
                batch_embeddings = np.vstack([text_embeddings, image_embeddings])
            elif len(text_embeddings) > 0:
                batch_embeddings = text_embeddings
            elif len(image_embeddings) > 0:
                batch_embeddings = image_embeddings
            else:
                continue  # Skip this batch if there are no embeddings

            if d is None:
                d = batch_embeddings.shape[1]
                index = self.faiss.index_factory(d, kwargs["index_type"], self.faiss.METRIC_INNER_PRODUCT)

            index.add(batch_embeddings)

            # Save intermediate results to avoid memory overflow
            with open(f"{index_dir}/docs_{batch_start}", "wb") as fp:
                pickle.dump(batch_docs, fp)
            with open(f"{index_dir}/vecs_{batch_start}", "wb") as fp:
                pickle.dump(batch_embeddings, fp)

        # Store final index
        self.faiss.write_index(index, f"{index_dir}/index")
        self.faiss_index = index
        self.index_dir = index_dir

    def base64_to_image(self, base64_string: str) -> Image.Image:
        """Convert a base64 string to a PIL Image."""
        image_data = base64.b64decode(base64_string.split(',')[1])
        return Image.open(BytesIO(image_data))


    def load_index(self, index_dir: str) -> None:
        self.index_dir = index_dir
        self.faiss_index = self.faiss.read_index(f"{index_dir}/index")
        self.docs = []
        self.vecs = []

        # Load documents and vectors in batches
        for file_name in sorted(os.listdir(index_dir)):
            if file_name.startswith("docs_"):
                with open(os.path.join(index_dir, file_name), "rb") as fp:
                    self.docs.extend(pickle.load(fp))
            elif file_name.startswith("vecs_"):
                with open(os.path.join(index_dir, file_name), "rb") as fp:
                    self.vecs.append(pickle.load(fp))

        self.vecs = np.vstack(self.vecs)

    @classmethod
    def get_vectors_from_index(cls, index_dir: str, ids: List[int]) -> List:
        vecs = []
        for file_name in sorted(os.listdir(index_dir)):
            if file_name.startswith("vecs_"):
                with open(os.path.join(index_dir, file_name), "rb") as fp:
                    batch_vecs = pickle.load(fp)
                    vecs.append(batch_vecs[ids])

        return np.vstack(vecs)

    def load_vecs(self, index_dir: str, ids: List[int]) -> List:
        """Loads vectors to the rm and returns them."""
        if self.vecs is None:
            self.vecs = self.get_vectors_from_index(index_dir, ids)
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
            embedded_queries = self.embed_text(queries, **kwargs)
        else:
            embedded_queries = queries

        distances, indices = self.faiss_index.search(embedded_queries, k)

        return distances, indices