import base64
from io import BytesIO
from typing import Callable

import pandas as pd
import qwen_vl_utils
from PIL import Image

import lotus


def cluster(col_name: str, ncentroids: int) -> Callable[[pd.DataFrame, int, bool], list[int]]:
    """
    Returns a function that clusters a DataFrame by a column using kmeans.

    Args:
        col_name (str): The column name to cluster by.
        ncentroids (int): The number of centroids to use.

    Returns:
        Callable: The function that clusters the DataFrame.
    """

    def ret(
        df: pd.DataFrame,
        niter: int = 20,
        verbose: bool = False,
    ) -> list[int]:
        import faiss

        """Cluster by column, and return a series in the dataframe with cluster-ids"""
        if col_name not in df.columns:
            raise ValueError(f"Column {col_name} not found in DataFrame")

        if ncentroids > len(df):
            raise ValueError(f"Number of centroids must be less than number of documents. {ncentroids} > {len(df)}")

        # get rmodel and index
        rm = lotus.settings.rm
        try:
            col_index_dir = df.attrs["index_dirs"][col_name]
        except KeyError:
            raise ValueError(f"Index directory for column {col_name} not found in DataFrame")

        if rm.index_dir != col_index_dir:
            rm.load_index(col_index_dir)
        assert rm.index_dir == col_index_dir

        ids = df.index.tolist()  # assumes df index hasn't been resest and corresponds to faiss index ids
        vec_set = rm.get_vectors_from_index(col_index_dir, ids)
        d = vec_set.shape[1]
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
        kmeans.train(vec_set)

        # get nearest centroid to each vector
        _, indices = kmeans.index.search(vec_set, 1)
        return list(map(int, indices.flatten().tolist()))

    return ret


def fetch_image(
    image: str | Image.Image | None, size_factor: int = 28, image_type: str = "Image"
) -> Image.Image | str | None:
    """
    Fetches an image from the internet or loads it from a file.

    Args:
        ele (str | Image.Image): The image URL or path.
        size_factor (int | None): The size factor to resize the image.
        image_type (str): The type of the element. Supported: Image or base64

    Returns:
        Image.Image: The image.
    """

    if image is None:
        return None

    assert image_type in ["Image", "base64"], f"image_type must be Image or base64, got {image_type}"

    pil_image = qwen_vl_utils.fetch_image({"image": image}, size_factor)
    if image_type == "base64":
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    return pil_image
