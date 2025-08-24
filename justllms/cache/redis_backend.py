"""Redis cache backend implementation."""

import asyncio
import json
import pickle
from typing import Any, Optional

try:
    import redis
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

import contextlib

from justllms.cache.backends import BaseCacheBackend


class RedisCacheBackend(BaseCacheBackend):
    """Redis-based cache backend for distributed caching."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "justllms:cache:",
        default_ttl: Optional[int] = None,
        connection_pool: Optional[redis.ConnectionPool] = None,
        redis_url: Optional[str] = None,
        serialize_method: str = "pickle",  # "pickle" or "json"
        **redis_kwargs: Any,
    ) -> None:
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis support requires the 'redis' package. Install with: pip install redis"
            )

        self.prefix = prefix
        self.default_ttl = default_ttl
        self.serialize_method = serialize_method

        # Initialize sync Redis client
        if redis_url:
            self.client = redis.from_url(redis_url, **redis_kwargs)
        elif connection_pool:
            self.client = redis.Redis(connection_pool=connection_pool)
        else:
            self.client = redis.Redis(
                host=host, port=port, db=db, password=password, **redis_kwargs
            )

        # Initialize async Redis client
        if redis_url:
            self.async_client = aioredis.from_url(redis_url, **redis_kwargs)
        else:
            self.async_client = aioredis.Redis(
                host=host, port=port, db=db, password=password, **redis_kwargs
            )

    def _make_key(self, key: str) -> str:
        """Add prefix to cache key."""
        return f"{self.prefix}{key}"

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        if self.serialize_method == "pickle":
            return pickle.dumps(value)
        elif self.serialize_method == "json":
            return json.dumps(value, default=str).encode("utf-8")
        else:
            raise ValueError(f"Unknown serialize_method: {self.serialize_method}")

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if self.serialize_method == "pickle":
            return pickle.loads(data)
        elif self.serialize_method == "json":
            return json.loads(data.decode("utf-8"))
        else:
            raise ValueError(f"Unknown serialize_method: {self.serialize_method}")

    def get(self, key: str) -> Optional[Any]:
        """Get a value from Redis cache."""
        try:
            redis_key = self._make_key(key)
            data = self.client.get(redis_key)

            if data is None:
                return None

            return self._deserialize(data)

        except Exception:
            # Log error but don't break the application
            # Redis cache get error - connection may be down
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in Redis cache."""
        try:
            redis_key = self._make_key(key)
            data = self._serialize(value)
            ttl = ttl or self.default_ttl

            if ttl:
                self.client.setex(redis_key, ttl, data)
            else:
                self.client.set(redis_key, data)

        except Exception:
            # Log error but don't break the application
            # Redis cache set error - connection may be down
            pass

    def delete(self, key: str) -> None:
        """Delete a value from Redis cache."""
        try:
            redis_key = self._make_key(key)
            self.client.delete(redis_key)

        except Exception:
            # Redis cache delete error - connection may be down
            pass

    def clear(self) -> None:
        """Clear all cache entries with the prefix."""
        try:
            pattern = f"{self.prefix}*"
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)

        except Exception:
            # Redis cache clear error - connection may be down
            pass

    def exists(self, key: str) -> bool:
        """Check if a key exists in Redis cache."""
        try:
            redis_key = self._make_key(key)
            return bool(self.client.exists(redis_key))

        except Exception:
            # Redis cache exists error - connection may be down
            pass
            return False

    # Async methods
    async def aget(self, key: str) -> Optional[Any]:
        """Async get from Redis cache."""
        try:
            redis_key = self._make_key(key)
            data = await self.async_client.get(redis_key)

            if data is None:
                return None

            return self._deserialize(data)

        except Exception:
            # Redis async cache get error - connection may be down
            pass
            return None

    async def aset(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Async set to Redis cache."""
        try:
            redis_key = self._make_key(key)
            data = self._serialize(value)
            ttl = ttl or self.default_ttl

            if ttl:
                await self.async_client.setex(redis_key, ttl, data)
            else:
                await self.async_client.set(redis_key, data)

        except Exception:
            # Redis async cache set error - connection may be down
            pass

    async def adelete(self, key: str) -> None:
        """Async delete from Redis cache."""
        try:
            redis_key = self._make_key(key)
            await self.async_client.delete(redis_key)

        except Exception:
            # Redis async cache delete error - connection may be down
            pass

    async def aclear(self) -> None:
        """Async clear all cache entries."""
        try:
            pattern = f"{self.prefix}*"
            keys = await self.async_client.keys(pattern)
            if keys:
                await self.async_client.delete(*keys)

        except Exception:
            # Redis async cache clear error - connection may be down
            pass

    async def aexists(self, key: str) -> bool:
        """Async check if key exists in Redis cache."""
        try:
            redis_key = self._make_key(key)
            result = await self.async_client.exists(redis_key)
            return bool(result)

        except Exception:
            # Redis async cache exists error - connection may be down
            pass
            return False

    def ping(self) -> bool:
        """Test Redis connection."""
        try:
            return self.client.ping()  # type: ignore
        except Exception:
            return False

    async def aping(self) -> bool:
        """Test async Redis connection."""
        try:
            return await self.async_client.ping()  # type: ignore
        except Exception:
            return False

    def info(self) -> dict:
        """Get Redis server info."""
        try:
            info = self.client.info()
            return {
                "redis_version": info.get("redis_version"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
            }
        except Exception as e:
            return {"error": str(e)}

    def cache_stats(self) -> dict:
        """Get cache statistics."""
        try:
            pattern = f"{self.prefix}*"
            keys = self.client.keys(pattern)

            return {
                "total_keys": len(keys),
                "prefix": self.prefix,
                "connection_status": "connected" if self.ping() else "disconnected",
            }
        except Exception as e:
            return {"error": str(e)}

    def close(self) -> None:
        """Close Redis connections."""
        with contextlib.suppress(Exception):
            self.client.close()

        with contextlib.suppress(Exception):
            asyncio.create_task(self.async_client.close())
