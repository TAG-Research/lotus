from typing import Callable
from io import BytesIO
from PIL import Image
import base64
import pandas as pd
import lotus
import requests


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

# function to convert an image to a base64 string with a prefix
def image_to_base64(image_path_or_url):
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        response = requests.get(image_path_or_url)
        image_data = response.content
    else:
        with open(image_path_or_url, "rb") as image_file:
            image_data = image_file.read()
    
    base64_image = base64.b64encode(image_data).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_image}"

# Function to decode base64 string to an image
def base64_to_image(base64_str):
    image_data = base64.b64decode(base64_str.split(',')[1])  # Remove prefix before decoding
    return Image.open(BytesIO(image_data))

def encode_images(df, column_name, new_column_name):
    df[new_column_name] = df[column_name].apply(image_to_base64)
    return df

@pd.api.extensions.register_dataframe_accessor("load_images")
class LoadImagesDataframe:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        
    @staticmethod
    def _validate(obj):
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")
        
    def __call__(self, col_name: str, new_col_name: str = 'image_base64') -> pd.DataFrame:
        """
        Load the images from the paths specified in col_name as base64 strings in new column

        Args:
            col_name (str): The column name to index.
            new_col_name (str, optional): The new column name to save the encoded images. Defaults to 'image_base64'.

        Returns:
            pd.DataFrame: The DataFrame with the index directory saved.
        """
        self._obj[new_col_name] = self._obj[col_name].apply(image_to_base64)
        return self._obj