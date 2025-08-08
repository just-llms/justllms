"""Cache manager for coordinating caching operations."""

import hashlib
import json
from typing import Any, Dict, List, Optional, Union

from justllms.cache.backends import BaseCacheBackend, DiskCacheBackend, InMemoryCacheBackend
from justllms.core.completion import CompletionResponse
from justllms.core.models import Message


class CacheManager:
    """Manages caching of LLM responses."""

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], Any]] = None,
        backend: Optional[BaseCacheBackend] = None,
        enabled: bool = True,
        ttl: Optional[int] = 3600,  # 1 hour default
        ignore_params: Optional[List[str]] = None,
    ):
        # Handle both dict and CacheConfig object
        if config is not None and hasattr(config, "model_dump"):
            # It's a Pydantic model, convert to dict
            self.config = config.model_dump()
        else:
            self.config = config or {}

        self.enabled = enabled and self.config.get("enabled", enabled)
        self.ttl = self.config.get("ttl", ttl)
        self.ignore_params = ignore_params or self.config.get("ignore_params", ["user", "seed"])

        # Initialize backend
        if backend:
            self.backend = backend
        else:
            self.backend = self._create_backend()

    def _create_backend(self) -> BaseCacheBackend:
        """Create a cache backend from config."""
        backend_type = self.config.get("backend", "memory")
        backend_config = self.config.get("backend_config", {})

        if backend_type == "memory":
            return InMemoryCacheBackend(
                max_size=backend_config.get("max_size", 1000),
                default_ttl=self.ttl,
            )
        elif backend_type == "disk":
            return DiskCacheBackend(
                cache_dir=backend_config.get("cache_dir"),
                size_limit=backend_config.get("size_limit", 1_000_000_000),
                default_ttl=self.ttl,
            )
        elif backend_type == "redis":
            from justllms.cache.redis_backend import RedisCacheBackend

            return RedisCacheBackend(
                host=backend_config.get("host", "localhost"),
                port=backend_config.get("port", 6379),
                db=backend_config.get("db", 0),
                password=backend_config.get("password"),
                prefix=backend_config.get("prefix", "justllms:cache:"),
                default_ttl=self.ttl,
                connection_pool=backend_config.get("connection_pool"),
                redis_url=backend_config.get("redis_url"),
                serialize_method=backend_config.get("serialize_method", "pickle"),
                **{
                    k: v
                    for k, v in backend_config.items()
                    if k
                    not in [
                        "host",
                        "port",
                        "db",
                        "password",
                        "prefix",
                        "connection_pool",
                        "redis_url",
                        "serialize_method",
                    ]
                },
            )
        else:
            raise ValueError(f"Unknown cache backend: {backend_type}")

    def get_cache_key(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a cache key for the request."""
        # Create a dictionary of all parameters
        cache_data = {
            "messages": [
                {
                    "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
                    "content": msg.content,
                }
                for msg in messages
            ],
            "model": model,
            "provider": provider,
        }

        # Add other parameters, excluding ignored ones
        for key, value in kwargs.items():
            if key not in self.ignore_params and value is not None:
                cache_data[key] = value

        # Sort keys for consistent hashing
        sorted_data = json.dumps(cache_data, sort_keys=True)

        # Generate hash
        return hashlib.sha256(sorted_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[CompletionResponse]:
        """Get a cached response."""
        if not self.enabled:
            return None

        cached_data = self.backend.get(key)
        if cached_data is None:
            return None

        # Reconstruct CompletionResponse
        if isinstance(cached_data, dict):
            # Mark as cached
            cached_data["cached"] = True

            # Reconstruct the response
            from justllms.core.models import Choice, Message, Usage

            choices = []
            for choice_data in cached_data.get("choices", []):
                msg_data = choice_data.get("message", {})
                message = Message(
                    role=msg_data.get("role", "assistant"),
                    content=msg_data.get("content", ""),
                )
                choice = Choice(
                    index=choice_data.get("index", 0),
                    message=message,
                    finish_reason=choice_data.get("finish_reason"),
                )
                choices.append(choice)

            usage_data = cached_data.get("usage")
            usage = None
            if usage_data:
                usage = Usage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                    estimated_cost=usage_data.get("estimated_cost"),
                )

            return CompletionResponse(
                id=cached_data.get("id", ""),
                model=cached_data.get("model", ""),
                choices=choices,
                usage=usage,
                created=cached_data.get("created"),
                system_fingerprint=cached_data.get("system_fingerprint"),
                provider=cached_data.get("provider"),
                cached=True,
            )

        return cached_data  # type: ignore

    def set(
        self,
        key: str,
        response: CompletionResponse,
        ttl: Optional[int] = None,
    ) -> None:
        """Cache a response."""
        if not self.enabled:
            return

        # Convert response to cacheable format
        cache_data = response.to_dict()

        self.backend.set(key, cache_data, ttl or self.ttl)

    async def aget(self, key: str) -> Optional[CompletionResponse]:
        """Async get a cached response."""
        if not self.enabled:
            return None

        cached_data = await self.backend.aget(key)
        if cached_data is None:
            return None

        # Use same reconstruction logic as sync get
        return self.get(key)

    async def aset(
        self,
        key: str,
        response: CompletionResponse,
        ttl: Optional[int] = None,
    ) -> None:
        """Async cache a response."""
        if not self.enabled:
            return

        # Convert response to cacheable format
        cache_data = response.to_dict()

        await self.backend.aset(key, cache_data, ttl or self.ttl)

    def delete(self, key: str) -> None:
        """Delete a cached response."""
        self.backend.delete(key)

    async def adelete(self, key: str) -> None:
        """Async delete a cached response."""
        await self.backend.adelete(key)

    def clear(self) -> None:
        """Clear all cached responses."""
        self.backend.clear()

    async def aclear(self) -> None:
        """Async clear all cached responses."""
        await self.backend.aclear()

    def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        return self.backend.exists(key)

    async def aexists(self, key: str) -> bool:
        """Async check if a key exists in cache."""
        return await self.backend.aexists(key)

    def enable(self) -> None:
        """Enable caching."""
        self.enabled = True

    def disable(self) -> None:
        """Disable caching."""
        self.enabled = False

    def is_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self.enabled
