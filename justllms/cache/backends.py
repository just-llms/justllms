"""Cache backend implementations."""

import asyncio
import contextlib
import json
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

try:
    import diskcache

    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False


class BaseCacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache with optional TTL (time-to-live) in seconds."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a value from cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        pass

    async def aget(self, key: str) -> Optional[Any]:
        """Async get (default implementation uses sync get)."""
        return await asyncio.get_event_loop().run_in_executor(None, self.get, key)

    async def aset(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Async set (default implementation uses sync set)."""
        await asyncio.get_event_loop().run_in_executor(None, self.set, key, value, ttl)

    async def adelete(self, key: str) -> None:
        """Async delete (default implementation uses sync delete)."""
        await asyncio.get_event_loop().run_in_executor(None, self.delete, key)

    async def aclear(self) -> None:
        """Async clear (default implementation uses sync clear)."""
        await asyncio.get_event_loop().run_in_executor(None, self.clear)

    async def aexists(self, key: str) -> bool:
        """Async exists (default implementation uses sync exists)."""
        return await asyncio.get_event_loop().run_in_executor(None, self.exists, key)


class InMemoryCacheBackend(BaseCacheBackend):
    """In-memory cache backend with LRU eviction."""

    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, tuple[Any, Optional[float]]] = OrderedDict()
        self._lock = asyncio.Lock()

    def _is_expired(self, expiry: Optional[float]) -> bool:
        """Check if an entry is expired."""
        if expiry is None:
            return False
        return time.time() > expiry

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is full."""
        while len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        if key not in self.cache:
            return None

        value, expiry = self.cache[key]

        if self._is_expired(expiry):
            del self.cache[key]
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache."""
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl if ttl else None

        self._evict_if_needed()
        self.cache[key] = (value, expiry)
        self.cache.move_to_end(key)

    def delete(self, key: str) -> None:
        """Delete a value from cache."""
        self.cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()

    def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        if key not in self.cache:
            return False

        _, expiry = self.cache[key]
        if self._is_expired(expiry):
            del self.cache[key]
            return False

        return True


class DiskCacheBackend(BaseCacheBackend):
    """Disk-based cache backend using diskcache library."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        size_limit: int = 1_000_000_000,  # 1GB default
        default_ttl: Optional[int] = None,
    ):
        if not DISKCACHE_AVAILABLE:
            raise ImportError(
                "Disk cache backend requires the 'diskcache' package. "
                "Install with: pip install diskcache"
            )

        self.cache_dir = cache_dir or Path.home() / ".justllms" / "cache"
        self.size_limit = size_limit
        self.default_ttl = default_ttl

        self.cache = diskcache.Cache(
            str(self.cache_dir),
            size_limit=size_limit,
        )

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        try:
            value = self.cache.get(key)
            if value is not None and isinstance(value, str) and value.startswith('{"'):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
            return value
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache."""
        ttl = ttl or self.default_ttl

        # Serialize complex objects to JSON
        if hasattr(value, "to_dict"):
            value = json.dumps(value.to_dict())
        elif isinstance(value, (dict, list)):
            value = json.dumps(value)

        self.cache.set(key, value, expire=ttl)

    def delete(self, key: str) -> None:
        """Delete a value from cache."""
        self.cache.delete(key)

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()

    def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        return key in self.cache

    def close(self) -> None:
        """Close the cache (cleanup)."""
        self.cache.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        with contextlib.suppress(Exception):
            self.close()
