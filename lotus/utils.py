from typing import Callable

import lotus


def cluster(col_name: str, ncentroids: int) -> Callable:
    """
    Returns a function that clusters a DataFrame by a column using kmeans.

    Args:
        col_name (str): The column name to cluster by.
        ncentroids (int): The number of centroids to use.

    Returns:
        Callable: The function that clusters the DataFrame.
    """

    def ret(
        df,
        niter: int = 20,
        verbose: bool = False,
    ):
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
        return indices.flatten()

    return ret
