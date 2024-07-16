import pandas as pd

import lotus


@pd.api.extensions.register_dataframe_accessor("sem_cluster_by")
class SemClusterByDataframe:
    """DataFrame accessor for semantic clustering."""

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def __call__(
        self,
        col_name: str,
        ncentroids: int,
        niter: int = 20,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Perform semantic clustering on the DataFrame.

        Args:
            col_name (str): The column name to cluster on.
            ncentroids (int): The number of centroids.
            niter (int): The number of iterations.
            verbose (bool): Whether to print verbose output.

        Returns:
            pd.DataFrame: The DataFrame with the cluster assignments.
        """
        cluster_fn = lotus.utils.cluster(col_name, ncentroids)
        indices = cluster_fn(
            self._obj,
            niter=niter,
            verbose=verbose,
        )

        self._obj["cluster_id"] = pd.Series(indices, index=self._obj.index)
        return self._obj
