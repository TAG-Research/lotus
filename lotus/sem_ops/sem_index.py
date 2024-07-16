import pandas as pd

import lotus


@pd.api.extensions.register_dataframe_accessor("sem_index")
class SemIndexDataframe:
    """DataFrame accessor for semantic indexing."""

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._obj.attrs["index_dirs"] = {}

    @staticmethod
    def _validate(obj):
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def __call__(self, col_name: str, index_dir: str) -> pd.DataFrame:
        """
        Index a column in the DataFrame.

        Args:
            col_name (str): The column name to index.
            index_dir (str): The directory to save the index.

        Returns:
            pd.DataFrame: The DataFrame with the index directory saved.
        """
        documents = self._obj[col_name].tolist()
        rm = lotus.settings.rm
        if rm is None:
            raise AttributeError("Must set rm in lotus.settings")
        rm.index(documents, index_dir)
        self._obj.attrs["index_dirs"][col_name] = index_dir
        return self._obj
