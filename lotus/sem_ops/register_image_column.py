from typing import Any

import pandas as pd

@pd.api.extensions.register_dataframe_accessor("register_image_column")
class RegisterImageColumnDataframe:
    """DataFrame accessor for registering image columns."""

    def __init__(self, pandas_obj: Any) -> None:
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._obj.attrs["image_columns"] = set()

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")
        
    def __call__(self, col_name: str) -> pd.DataFrame:
        """
        Register a column in the DataFrame as an image column.

        Args:
            col_name (str): The column name to register as an image column.
            image_dir (str): The directory where the images are saved.

        Returns:
            pd.DataFrame: The DataFrame with the image directory saved.
        """
        self._obj.attrs["image_columns"].add(col_name)
        return self._obj