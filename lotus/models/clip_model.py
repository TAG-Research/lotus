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

from lotus.templates import task_instructions

class CLIPModelRetriever(RM):
    """CLIP retriever model with multimodal (text & image) embedding support"""

    def __init__(
        self, 
        model: str = "openai/clip-vit-base-patch32", 
        device: Optional[str] = None, 
        batch_size: Optional[int] = 5000,
        similarity_weights: Optional[list] = None,
        **kwargs
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model)
        self.model = CLIPModel.from_pretrained(model).to(device)
        
        # Fixed weights for combining similarities based on empirical performance
        # These default weights prioritize text-text and image-image direct matches
        # while still considering cross-modal similarities
        
        if similarity_weights is None:
            similarity_weights = [0.4, 0.4, 0.1, 0.1] # [text-text, image-image, text-image, image-text]
            
        self.similarity_weights = torch.tensor(similarity_weights, device=device)
    
        self.faiss_index = None
        self.index_dir = None
        self.docs = None
        self.kwargs = {"normalize": True, "index_type": "Flat", **kwargs}
        self.batch_size = batch_size
        self.vecs = None

        import faiss
        self.faiss = faiss
        
    def create_combined_embedding(
        self,
        text: str,
        images: List[str],
        **kwargs: Dict[str, Any]
    ) -> np.ndarray:
        """Create a combined embedding using both text and images."""
        with torch.no_grad():
            # Get text embeddings
            text_inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            text_features = self.model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, p=2, dim=1)

            # Get image embeddings
            image_features_list = []
            for img_str in images:
                img = self.base64_to_image(img_str)
                image_inputs = self.processor(
                    images=img,
                    return_tensors="pt"
                ).to(self.device)
                image_features = self.model.get_image_features(**image_inputs)
                image_features = F.normalize(image_features, p=2, dim=1)
                image_features_list.append(image_features)

            # Average multiple image features if present
            if image_features_list:
                image_features = torch.mean(torch.stack(image_features_list), dim=0)
            else:
                image_features = torch.zeros_like(text_features)
                
            # Calculate combined features
            combined_features = (
                self.similarity_weights[0] * text_features +
                self.similarity_weights[1] * image_features
            )

            # Add cross-modal terms only if both text and images are present
            if images:
                # Calculate cosine similarity for cross-modal terms
                text_image_sim = (text_features @ image_features.T).diagonal().unsqueeze(1)
                image_text_sim = text_image_sim  # symmetric in this case
                
                combined_features += (
                    self.similarity_weights[2] * text_image_sim * text_features +
                    self.similarity_weights[3] * image_text_sim * image_features
                )

            if kwargs.get("normalize", True):
                combined_features = F.normalize(combined_features, p=2, dim=1)

            return combined_features.cpu().numpy()

    def embed_text(self, texts: List[str], **kwargs: Dict[str, Any]) -> np.ndarray:
        """Run the text embedding model."""
        kwargs = {**self.kwargs, **kwargs}
        batch_size = kwargs.get("batch_size", self.batch_size)
        embeddings = []

        for i, batch_start in enumerate(tqdm(range(0, len(texts), batch_size))):
            batch = texts[batch_start : batch_start + batch_size]
            with torch.no_grad():
                inputs = self.processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                outputs = self.model.get_text_features(**inputs)
                embeddings.append(outputs)

        embeddings = torch.cat(embeddings, dim=0)
        if kwargs["normalize"]:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def embed_images(self, images: List[Any], **kwargs: Dict[str, Any]) -> np.ndarray:
        """Run the image embedding model."""

        kwargs = {**self.kwargs, **kwargs}
        
        # Check if images are base64 strings and convert to PIL images
        for i, img in enumerate(images):
            if isinstance(img, str) and img.startswith("data:image"):
                images[i] = self.base64_to_image(img)

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
        os.makedirs(index_dir, exist_ok=True)
        kwargs = {**self.kwargs, **kwargs}
        d = None
        index = None

        for batch_start in tqdm(range(0, len(docs), self.batch_size)):
            batch_docs = docs[batch_start:batch_start + self.batch_size]
            batch_embeddings = []

            for doc in batch_docs:
                # Extract images and clean text using the provided function
                images, clean_text = task_instructions.extract_image_data(doc)
                
                if images and (isinstance(clean_text, str) and clean_text.strip() != ""):
                    # Create combined embedding for documents with images
                    embedding = self.create_combined_embedding(clean_text, images, **kwargs)
                elif images:
                    # Create image-only embedding for documents without text
                    embedding = self.embed_images(images, **kwargs)
                else:
                    # Create text-only embedding for documents without images
                    embedding = self.embed_text([clean_text], **kwargs)
                
                batch_embeddings.append(embedding)

            batch_embeddings = np.vstack(batch_embeddings)

            if d is None:
                d = batch_embeddings.shape[1]
                index = self.faiss.index_factory(
                    d,
                    kwargs["index_type"],
                    self.faiss.METRIC_INNER_PRODUCT
                )

            index.add(batch_embeddings)

            # Save intermediate results
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
        """Modified to handle both text and image queries"""
        if isinstance(queries, str):
            queries = [queries]
            
        if not isinstance(queries[0], str):
            embedded_queries = queries
        else:         
            embedded_queries = []
            for query in queries:
                # Extract any images from the query text
                images, clean_text = task_instructions.extract_image_data(query)
            
                # check if clean_text is a valid string, and if it is empty
                # check if images is not empty
                if images and (isinstance(clean_text, str) and clean_text.strip() != ""):
                    # If query contains images & text, use combined embedding
                    embedding = self.create_combined_embedding(clean_text, images, **kwargs)
                elif images:
                    # If query is image-only, use image embedding
                    embedding = self.embed_images(images, **kwargs)
                else:
                    # If query is text-only, use text embedding
                    embedding = self.embed_text([clean_text], **kwargs)
                
                embedded_queries.append(embedding)


            # Stack all query embeddings
            embedded_queries = np.vstack(embedded_queries)

        # Search using the appropriate embeddings
        distances, indices = self.faiss_index.search(embedded_queries, k)

        return distances, indices