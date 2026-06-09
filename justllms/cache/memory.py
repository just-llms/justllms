"""Thread-safe in-memory LRU cache backend."""

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from justllms.cache.base import BaseCacheBackend


class InMemoryCache(BaseCacheBackend):
    """In-process LRU cache with optional per-entry TTL.

    Entries are evicted least-recently-used first once ``max_size`` is
    reached. Expired entries are treated as misses and purged lazily.

    Args:
        max_size: Maximum number of entries to keep (default 1000).
    """

    def __init__(self, max_size: int = 1000) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer")
        self.max_size = max_size
        # key -> (value, expires_at). expires_at is monotonic time or None.
        self._data: OrderedDict[str, Tuple[Dict[str, Any], Optional[float]]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached value, honoring TTL and updating LRU order."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return None
            value, expires_at = entry
            if expires_at is not None and time.monotonic() >= expires_at:
                # Expired: purge lazily and report a miss.
                del self._data[key]
                self._misses += 1
                return None
            self._data.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Dict[str, Any], ttl: Optional[float] = None) -> None:
        """Store a value, evicting least-recently-used entries if full."""
        expires_at = time.monotonic() + ttl if ttl is not None else None
        with self._lock:
            if key in self._data:
                del self._data[key]
            self._data[key] = (value, expires_at)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    def delete(self, key: str) -> None:
        """Remove a single entry if present."""
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    @property
    def stats(self) -> Dict[str, Any]:
        """Return hit/miss counters and current size."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._data),
                "max_size": self.max_size,
            }
