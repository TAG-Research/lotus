from lotus.dtype_extensions.image import ImageDtype, ImageArray
import pandas as pd

pd.api.extensions.register_extension_dtype(ImageDtype)
__all__ = ["ImageDtype", "ImageArray"]