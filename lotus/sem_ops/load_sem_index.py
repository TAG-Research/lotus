from typing import Any

import pandas as pd


@pd.api.extensions.register_dataframe_accessor("load_sem_index")
class LoadSemIndexDataframe:
    """DataFrame accessor for loading a semantic index."""

    def __init__(self, pandas_obj: Any):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._obj.attrs["index_dirs"] = {}

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def __call__(self, col_name: str, index_dir: str) -> pd.DataFrame:
        """
        Load a semantic index for a column in the DataFrame.

        Args:
            col_name (str): The column name to load the index for.
            index_dir (str): The directory to load the index from.

        Returns:
            pd.DataFrame: The DataFrame with the index loaded.
        """
        self._obj.attrs["index_dirs"][col_name] = index_dir
        return self._obj
