from pandas.api.extensions import ExtensionDtype, ExtensionArray
import numpy as np
from PIL import Image
import io

class ImageDtype(ExtensionDtype):
    """Custom dtype for  Images."""
    name = 'image'
    type = Image.Image
    na_value = None
    
    @classmethod
    def construct_array_type(cls):
        return ImageArray

class ImageArray(ExtensionArray):
    """ExtensionArray for storing Images."""
    
    def __init__(self, values):
        """Initialize array with validation."""
        values = self._validate_values(values)
        self._data = np.array(values, dtype=object)
        self._dtype = ImageDtype()
    
    @staticmethod
    def _validate_values(values):
        """Validate that all values are Images or None."""
        if isinstance(values, (ImageArray, np.ndarray)):
            values = values.tolist()
        
        validated = []
        for i, val in enumerate(values):
            if val is None:
                validated.append(None)
            elif isinstance(val, Image.Image):
                validated.append(val)
            else:
                raise TypeError(
                    f"Value at index {i} has type {type(val).__name__}. "
                    "Expected .Image.Image or None."
                )
        return validated
    
    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """Create ImageArray from sequence of scalars."""
        scalars = cls._validate_values(scalars)
        if copy:
            scalars = [img.copy() if img is not None else None for img in scalars]
        return cls(scalars)
    
    @classmethod
    def _from_factorized(cls, values, original):
        """Create ImageArray from factorized values."""
        return original
    
    def __getitem__(self, item):
        """Get item(s) from array."""
        result = self._data[item]
        if isinstance(item, (int, np.integer)):
            return result
        return ImageArray(result)
    
    def __len__(self):
        """Length of array."""
        return len(self._data)
    
    def __eq__(self, other):
        """Equality comparison."""
        if isinstance(other, ImageArray):
            return np.array([
                _compare_images(img1, img2) 
                for img1, img2 in zip(self._data, other._data)
            ])
        elif isinstance(other, (Image.Image, type(None))):
            return np.array([
                _compare_images(img, other) 
                for img in self._data
            ])
        return NotImplemented
    
    def __setitem__(self, key, value):
        """Set item(s) in array with validation."""
        if isinstance(key, (int, np.integer)):
            if not (isinstance(value, Image.Image) or value is None):
                raise TypeError(
                    f"Cannot set value of type {type(value).__name__}. "
                    "Expected Image.Image or None."
                )
            self._data[key] = value
        else:
            value = self._validate_values(value)
            self._data[key] = value
    
    @property
    def dtype(self):
        """Return the dtype object."""
        return self._dtype
    
    @property
    def nbytes(self):
        """Return number of bytes in memory."""
        return sum(
            len(img_to_bytes(img)) if img is not None else 0
            for img in self._data
        )
    
    def isna(self):
        """Return boolean array indicating missing values."""
        return np.array([img is None for img in self._data])
    
    def take(self, indexer, allow_fill=False, fill_value=None):
        """Take elements from array."""
        if allow_fill:
            if fill_value is not None and not (isinstance(fill_value, Image.Image) or fill_value is None):
                raise TypeError(
                    f"Fill value must be Image.Image or None, not {type(fill_value).__name__}"
                )
            if fill_value is None:
                fill_value = self.dtype.na_value
            
            result = np.array([
                self._data[idx] if idx >= 0 else fill_value
                for idx in indexer
            ])
        else:
            result = self._data.take(indexer)
        
        return ImageArray(result)
    
    def copy(self):
        """Return deep copy of array."""
        return ImageArray([
            img.copy() if img is not None else None 
            for img in self._data
        ])
    
    @classmethod
    def _concat_same_type(cls, to_concat):
        """Concatenate multiple arrays."""
        return cls(np.concatenate([array._data for array in to_concat]))
    
    def interpolate(self, method='linear', axis=0, limit=None, inplace=False,
                   limit_direction=None, limit_area=None, downcast=None, **kwargs):
        """Interpolate missing values."""
        return self.copy() if not inplace else self

def _compare_images(img1, img2):
    """Compare two Images for equality."""
    if img1 is None and img2 is None:
        return True
    if img1 is None or img2 is None:
        return False
    if img1.size != img2.size:
        return False
    if img1.mode != img2.mode:
        return False
    return np.array_equal(np.array(img1), np.array(img2))

def img_to_bytes(img):
    """Convert Image to bytes."""
    if img is None:
        return None
    buf = io.BytesIO()
    img.save(buf, format=img.format or 'PNG')
    return buf.getvalue()

def bytes_to_img(b):
    """Convert bytes to Image."""
    if b is None:
        return None
    return Image.open(io.BytesIO(b))