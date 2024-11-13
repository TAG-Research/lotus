import sys
from typing import Sequence, Union

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype
from PIL import Image

from lotus.utils import fetch_image


class ImageDtype(ExtensionDtype):
    name = "image"
    type = Image.Image
    na_value = None

    @classmethod
    def construct_array_type(cls):
        return ImageArray


class ImageArray(ExtensionArray):
    def __init__(self, values: np.ndarray):
        self._data = np.asarray(values, dtype=object)
        self._dtype = ImageDtype()

    def __getitem__(self, item: int | slice | Sequence[int]) -> Union[Image.Image, None, "ImageArray"]:
        result = self._data[item]
        if isinstance(item, (int, np.integer)):
            assert result is None or isinstance(result, (Image.Image, str))
            image_result = fetch_image(result)
            assert (
                isinstance(image_result, Image.Image) or image_result is None
            ), f"Expected Image.Image or None, got {type(image_result)}"
            return image_result
        return ImageArray(result)

    def isna(self) -> np.ndarray:
        return pd.isna(self._data)

    def take(self, indices: Sequence[int], allow_fill: bool = False, fill_value=None) -> "ImageArray":
        result = self._data.take(indices)
        if allow_fill and fill_value is not None:
            result[indices == -1] = fill_value
        return ImageArray(result)

    def copy(self) -> "ImageArray":
        return ImageArray(self._data.copy())

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        if copy:
            scalars = np.array(scalars, dtype=object, copy=True)
        return cls(scalars, lambda x: x)  # Default identity transform

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> np.ndarray:  # type: ignore
        # check if other is iterable
        if isinstance(other, ImageArray):
            return np.array([_compare_images(img1, img2) for img1, img2 in zip(self._data, other._data)], dtype=bool)

        if hasattr(other, "__iter__") and not isinstance(other, str):
            if len(other) != len(self):
                return np.repeat(False, len(self))
            return np.array([_compare_images(img1, img2) for img1, img2 in zip(self._data, other)], dtype=bool)
        return np.array([_compare_images(img, other) for img in self._data], dtype=bool)

    @property
    def dtype(self) -> ImageDtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return sum(sys.getsizeof(img) for img in self._data if img)

    def __repr__(self) -> str:
        return f"ImageArray([{', '.join(['<Image>' if img is not None else 'None' for img in self._data[:5]])}, ...])"

    def _formatter(self, boxed: bool = False):
        return lambda x: "<Image>" if x is not None else "None"


def _compare_images(img1, img2) -> bool:
    if img1 is None or img2 is None:
        return img1 is img2
    if isinstance(img1, Image.Image) or isinstance(img2, Image.Image):
        img1 = fetch_image(img1)
        img2 = fetch_image(img2)
        return img1.size == img2.size and img1.mode == img2.mode and img1.tobytes() == img2.tobytes()
    else:
        return img1 == img2
