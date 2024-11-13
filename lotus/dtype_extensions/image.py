import sys
from typing import Optional, Sequence, Union

import numpy as np
from pandas.api.extensions import ExtensionArray, ExtensionDtype
from PIL import Image

from lotus.utils import fetch_image


class ImageDtype(ExtensionDtype):
    name = 'image'
    type = Image.Image
    na_value = None

    @classmethod
    def construct_array_type(cls):
        return ImageArray


class ImageArray(ExtensionArray):
    def __init__(self, values: Sequence[Optional[Image.Image]]):
        self._data = values
        self._dtype = ImageDtype()

    def __getitem__(self, item: Union[int, slice, Sequence[int]]) -> Union[Image.Image, 'ImageArray']:
        if isinstance(item, (int, np.integer)):
            return fetch_image(self._data[item])
        if isinstance(item, slice):
            return ImageArray(self._data[item])
        return ImageArray([self._data[i] for i in item])

    def __setitem__(self, key: Union[int, slice, Sequence[int]], value: Union[Image.Image, Sequence[Image.Image]]):
        if isinstance(key, (int, np.integer)):
            self._data[key] = value
        else:
            for i, k in enumerate(key):
                self._data[k] = value[i]

    def isna(self) -> np.ndarray:
        return np.array([img is None for img in self._data], dtype=bool)

    def take(self, indexer: Sequence[int], allow_fill: bool = False, fill_value: Optional[Image.Image] = None) -> 'ImageArray':
        if allow_fill:
            fill_value = fill_value if fill_value is not None else self.dtype.na_value
            result = [self._data[idx] if idx >= 0 else fill_value for idx in indexer]
        else:
            result = [self._data[idx] for idx in indexer]
        return ImageArray(result)

    def copy(self) -> 'ImageArray':
        return ImageArray([img.copy() if img and hasattr(img, "copy") else img for img in self._data])

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls([img.copy() if img and copy  and hasattr(img, 'copy') else img for img in scalars])

    @classmethod
    def _from_factorized(cls, values, original):
        return original

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence['ImageArray']) -> 'ImageArray':
        combined = [img for array in to_concat for img in array._data]
        return cls(combined)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> np.ndarray:
        # check if other is iterable
        if isinstance(other, ImageArray):
            return np.array([_compare_images(img1, img2) for img1, img2 in zip(self._data, other._data)], dtype=bool)
        if hasattr(other, '__iter__') and not isinstance(other, str):
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
        return lambda x: '<Image>' if x is not None else 'None'


def _compare_images(img1: Optional[Image.Image], img2: Optional[Image.Image]) -> bool:
    if img1 is None or img2 is None:
        return img1 is img2
    if isinstance(img1, Image.Image) or isinstance(img2, Image.Image):
        img1 = fetch_image(img1)
        img2 = fetch_image(img2)
        return (img1.size == img2.size and img1.mode == img2.mode and img1.tobytes() == img2.tobytes())
    else:
        return img1 == img2
