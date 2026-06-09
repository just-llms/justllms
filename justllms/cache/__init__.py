"""Response caching for justllms.

Provides exact-match caching of non-streaming, non-tool completion
responses, keyed on provider, model, messages, and generation parameters.
"""

from justllms.cache.base import BaseCacheBackend
from justllms.cache.disk import DiskCache
from justllms.cache.manager import (
    ResponseCache,
    _response_from_cache_dict,
    _response_to_cache_dict,
    make_cache_key,
)
from justllms.cache.memory import InMemoryCache

__all__ = [
    "BaseCacheBackend",
    "DiskCache",
    "InMemoryCache",
    "ResponseCache",
    "make_cache_key",
    "_response_from_cache_dict",
    "_response_to_cache_dict",
]
