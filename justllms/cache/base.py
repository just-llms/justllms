"""Abstract base class for response cache backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseCacheBackend(ABC):
    """Abstract interface for cache storage backends.

    Implementations store JSON-serializable response dictionaries keyed by
    opaque string keys, with optional per-entry time-to-live expiry.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached value by key.

        Args:
            key: Cache key to look up.

        Returns:
            The cached dictionary, or None if the key is missing or expired.
        """

    @abstractmethod
    def set(self, key: str, value: Dict[str, Any], ttl: Optional[float] = None) -> None:
        """Store a value under the given key.

        Args:
            key: Cache key to store under.
            value: JSON-serializable dictionary to cache.
            ttl: Optional time-to-live in seconds. None means no expiry.
        """

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove a single entry from the cache.

        Args:
            key: Cache key to remove. Missing keys are ignored.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries from the cache."""

    @property
    def stats(self) -> Dict[str, Any]:
        """Return backend statistics (hits, misses, size) when available."""
        return {}
