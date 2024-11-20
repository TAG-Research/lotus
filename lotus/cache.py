from collections import OrderedDict
from functools import wraps
from typing import Any, Callable

import lotus


def require_cache_enabled(func: Callable) -> Callable:
    """Decorator to check if caching is enabled before calling the function."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not lotus.settings.enable_cache:
            return None
        return func(self, *args, **kwargs)

    return wrapper


class Cache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: OrderedDict[str, Any] = OrderedDict()

    @require_cache_enabled
    def get(self, key: str) -> Any | None:
        if key in self.cache:
            lotus.logger.debug(f"Cache hit for {key}")

        return self.cache.get(key)

    @require_cache_enabled
    def insert(self, key: str, value: Any):
        self.cache[key] = value

        # LRU eviction
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def reset(self, max_size: int | None = None):
        self.cache.clear()
        if max_size is not None:
            self.max_size = max_size
