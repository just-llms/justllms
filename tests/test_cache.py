"""Tests for the response caching feature."""

import threading
import time
from typing import Any, Dict, List, Optional

import pytest

from justllms.cache import DiskCache, InMemoryCache, make_cache_key
from justllms.core.base import BaseProvider, BaseResponse
from justllms.core.client import Client
from justllms.core.models import (
    Choice,
    Message,
    ModelInfo,
    ProviderConfig,
    Role,
    Usage,
)
from justllms.exceptions import ConfigurationError


def _msg(content: str, role: Role = Role.USER) -> Message:
    return Message(role=role, content=content)


# ---------------------------------------------------------------------------
# Cache key generation
# ---------------------------------------------------------------------------


class TestMakeCacheKey:
    def test_deterministic(self):
        messages = [_msg("hello")]
        params = {"temperature": 0.5, "max_tokens": 10}
        key1 = make_cache_key("openai", "gpt-4", messages, params)
        key2 = make_cache_key("openai", "gpt-4", [_msg("hello")], dict(params))
        assert key1 == key2
        assert len(key1) == 64  # sha256 hex digest

    def test_param_order_independent(self):
        messages = [_msg("hello")]
        key1 = make_cache_key("p", "m", messages, {"temperature": 0.5, "max_tokens": 10})
        key2 = make_cache_key("p", "m", messages, {"max_tokens": 10, "temperature": 0.5})
        assert key1 == key2

    def test_sensitive_to_temperature(self):
        messages = [_msg("hello")]
        key1 = make_cache_key("p", "m", messages, {"temperature": 0.5})
        key2 = make_cache_key("p", "m", messages, {"temperature": 0.7})
        assert key1 != key2

    def test_sensitive_to_model_and_provider(self):
        messages = [_msg("hello")]
        params: Dict[str, Any] = {}
        assert make_cache_key("p", "m1", messages, params) != make_cache_key(
            "p", "m2", messages, params
        )
        assert make_cache_key("p1", "m", messages, params) != make_cache_key(
            "p2", "m", messages, params
        )

    def test_sensitive_to_messages(self):
        key1 = make_cache_key("p", "m", [_msg("hello")], {})
        key2 = make_cache_key("p", "m", [_msg("goodbye")], {})
        key3 = make_cache_key("p", "m", [_msg("hello", role=Role.SYSTEM)], {})
        assert len({key1, key2, key3}) == 3

    def test_cache_control_keys_ignored(self):
        messages = [_msg("hello")]
        key1 = make_cache_key("p", "m", messages, {"temperature": 0.5})
        key2 = make_cache_key(
            "p",
            "m",
            messages,
            {"temperature": 0.5, "cache": True, "cache_ttl": 5.0, "timeout": 30.0},
        )
        assert key1 == key2

    def test_none_params_ignored(self):
        messages = [_msg("hello")]
        key1 = make_cache_key("p", "m", messages, {"temperature": 0.5})
        key2 = make_cache_key("p", "m", messages, {"temperature": 0.5, "seed": None})
        assert key1 == key2


# ---------------------------------------------------------------------------
# InMemoryCache
# ---------------------------------------------------------------------------


class TestInMemoryCache:
    def test_set_get_delete_clear(self):
        cache = InMemoryCache(max_size=10)
        assert cache.get("k") is None
        cache.set("k", {"v": 1})
        assert cache.get("k") == {"v": 1}
        cache.delete("k")
        assert cache.get("k") is None
        cache.set("a", {"v": 1})
        cache.set("b", {"v": 2})
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_ttl_expiry(self):
        cache = InMemoryCache()
        cache.set("k", {"v": 1}, ttl=0.05)
        assert cache.get("k") == {"v": 1}
        time.sleep(0.08)
        assert cache.get("k") is None
        assert len(cache) == 0  # purged lazily

    def test_no_ttl_means_no_expiry(self):
        cache = InMemoryCache()
        cache.set("k", {"v": 1}, ttl=None)
        time.sleep(0.02)
        assert cache.get("k") == {"v": 1}

    def test_lru_eviction(self):
        cache = InMemoryCache(max_size=3)
        cache.set("a", {"v": 1})
        cache.set("b", {"v": 2})
        cache.set("c", {"v": 3})
        # Touch "a" so "b" becomes least recently used.
        assert cache.get("a") is not None
        cache.set("d", {"v": 4})
        assert cache.get("b") is None
        assert cache.get("a") is not None
        assert cache.get("c") is not None
        assert cache.get("d") is not None
        assert len(cache) == 3

    def test_invalid_max_size(self):
        with pytest.raises(ValueError):
            InMemoryCache(max_size=0)

    def test_stats(self):
        cache = InMemoryCache()
        cache.set("k", {"v": 1})
        cache.get("k")
        cache.get("missing")
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_thread_safety_smoke(self):
        cache = InMemoryCache(max_size=50)
        errors: List[Exception] = []

        def worker(worker_id: int) -> None:
            try:
                for i in range(200):
                    key = f"k-{worker_id}-{i % 10}"
                    cache.set(key, {"v": i})
                    cache.get(key)
                    if i % 7 == 0:
                        cache.delete(key)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(cache) <= 50


# ---------------------------------------------------------------------------
# DiskCache
# ---------------------------------------------------------------------------


class TestDiskCache:
    def test_roundtrip(self, tmp_path):
        cache = DiskCache(path=tmp_path / "cache.db")
        assert cache.get("k") is None
        cache.set("k", {"v": [1, 2], "nested": {"a": True}})
        assert cache.get("k") == {"v": [1, 2], "nested": {"a": True}}
        cache.delete("k")
        assert cache.get("k") is None
        cache.set("a", {"v": 1})
        cache.clear()
        assert cache.get("a") is None
        cache.close()

    def test_ttl_expiry(self, tmp_path):
        cache = DiskCache(path=tmp_path / "cache.db")
        cache.set("k", {"v": 1}, ttl=0.05)
        assert cache.get("k") == {"v": 1}
        time.sleep(0.08)
        assert cache.get("k") is None
        cache.close()

    def test_persistence_across_instances(self, tmp_path):
        db_path = tmp_path / "cache.db"
        cache1 = DiskCache(path=db_path)
        cache1.set("k", {"v": "persisted"})
        cache1.close()

        cache2 = DiskCache(path=db_path)
        assert cache2.get("k") == {"v": "persisted"}
        cache2.close()

    def test_stats(self, tmp_path):
        cache = DiskCache(path=tmp_path / "cache.db")
        cache.set("k", {"v": 1})
        cache.get("k")
        cache.get("missing")
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        cache.close()


# ---------------------------------------------------------------------------
# Client integration
# ---------------------------------------------------------------------------


class FakeProvider(BaseProvider):
    """Minimal provider returning canned responses and counting calls."""

    requires_api_key = False

    def __init__(self) -> None:
        super().__init__(ProviderConfig(name="fake"))
        self.call_count = 0
        self.last_kwargs: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "fake"

    def get_available_models(self) -> Dict[str, ModelInfo]:
        return {
            "fake-model": ModelInfo(
                name="fake-model",
                provider="fake",
                cost_per_1k_prompt_tokens=0.001,
                cost_per_1k_completion_tokens=0.002,
            )
        }

    def complete(
        self,
        messages: List[Message],
        model: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> BaseResponse:
        self.call_count += 1
        self.last_kwargs = dict(kwargs)
        message = Message(role=Role.ASSISTANT, content=f"response-{self.call_count}")
        choice = Choice(index=0, message=message, finish_reason="stop")
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        return BaseResponse(
            id=f"resp-{self.call_count}",
            model=model,
            choices=[choice],
            usage=usage,
            created=1700000000,
        )


def _make_client(cache_config: Optional[Dict[str, Any]] = None) -> "tuple[Client, FakeProvider]":
    provider = FakeProvider()
    config: Dict[str, Any] = {"providers": {}}
    if cache_config is not None:
        config["cache"] = cache_config
    client = Client(config=config, providers={"fake": provider})
    return client, provider


class TestClientCaching:
    def test_cache_hit_on_identical_request(self):
        client, provider = _make_client({"enabled": True})
        messages = [{"role": "user", "content": "hello"}]

        first = client.completion.create(messages=messages, provider="fake")
        second = client.completion.create(messages=messages, provider="fake")

        assert provider.call_count == 1
        assert first.cached is False
        assert second.cached is True
        assert second.content == first.content
        assert second.provider == "fake"
        assert second.model == first.model
        assert second.usage is not None and first.usage is not None
        assert second.usage.total_tokens == first.usage.total_tokens
        # Cost is estimated before caching, so it round-trips too.
        assert first.usage.estimated_cost == pytest.approx(0.001 * 10 / 1000 + 0.002 * 5 / 1000)
        assert second.usage.estimated_cost == first.usage.estimated_cost

    def test_cache_hit_via_routed_path(self):
        client, provider = _make_client({"enabled": True})
        messages = [{"role": "user", "content": "routed"}]

        first = client.completion.create(messages=messages)
        second = client.completion.create(messages=messages)

        assert provider.call_count == 1
        assert second.cached is True
        assert second.content == first.content

    def test_different_prompt_misses(self):
        client, provider = _make_client({"enabled": True})
        client.completion.create(messages=[{"role": "user", "content": "one"}], provider="fake")
        client.completion.create(messages=[{"role": "user", "content": "two"}], provider="fake")
        assert provider.call_count == 2

    def test_different_params_miss(self):
        client, provider = _make_client({"enabled": True})
        messages = [{"role": "user", "content": "hello"}]
        client.completion.create(messages=messages, provider="fake", temperature=0.1)
        client.completion.create(messages=messages, provider="fake", temperature=0.9)
        assert provider.call_count == 2

    def test_cache_false_bypasses(self):
        client, provider = _make_client({"enabled": True})
        messages = [{"role": "user", "content": "hello"}]

        client.completion.create(messages=messages, provider="fake", cache=False)
        client.completion.create(messages=messages, provider="fake", cache=False)
        assert provider.call_count == 2

        # Bypassed calls also did not write to the cache.
        response = client.completion.create(messages=messages, provider="fake")
        assert provider.call_count == 3
        assert response.cached is False

    def test_cache_kwargs_not_forwarded_to_provider(self):
        client, provider = _make_client({"enabled": True})
        client.completion.create(
            messages=[{"role": "user", "content": "hello"}],
            provider="fake",
            cache=True,
            cache_ttl=60.0,
            temperature=0.2,
        )
        assert "cache" not in provider.last_kwargs
        assert "cache_ttl" not in provider.last_kwargs
        assert provider.last_kwargs.get("temperature") == 0.2

    def test_per_request_cache_ttl(self):
        client, provider = _make_client({"enabled": True, "ttl": None})
        messages = [{"role": "user", "content": "hello"}]

        client.completion.create(messages=messages, provider="fake", cache_ttl=0.05)
        time.sleep(0.08)
        client.completion.create(messages=messages, provider="fake")
        assert provider.call_count == 2

    def test_disabled_by_default(self):
        client, provider = _make_client()
        assert client.cache is None
        messages = [{"role": "user", "content": "hello"}]
        client.completion.create(messages=messages, provider="fake")
        client.completion.create(messages=messages, provider="fake")
        assert provider.call_count == 2

    def test_cache_true_without_config_raises(self):
        client, _ = _make_client()
        with pytest.raises(ConfigurationError, match="not configured"):
            client.completion.create(
                messages=[{"role": "user", "content": "hello"}],
                provider="fake",
                cache=True,
            )

    def test_unknown_backend_raises(self):
        with pytest.raises(ConfigurationError, match="Unknown cache backend"):
            _make_client({"enabled": True, "backend": "redis"})

    def test_disk_backend_via_config(self, tmp_path):
        db_path = str(tmp_path / "cache.db")
        client, provider = _make_client({"enabled": True, "backend": "disk", "path": db_path})
        messages = [{"role": "user", "content": "hello"}]

        first = client.completion.create(messages=messages, provider="fake")
        second = client.completion.create(messages=messages, provider="fake")
        assert provider.call_count == 1
        assert second.cached is True
        assert second.content == first.content

    def test_streaming_never_touches_cache(self):
        client, provider = _make_client({"enabled": True})
        assert client.cache is not None
        messages = [{"role": "user", "content": "hello"}]

        with pytest.raises(Exception):
            client.completion.create(messages=messages, provider="fake", stream=True)

        stats = client.cache.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0
        assert provider.call_count == 0

    def test_tools_never_touch_cache(self):
        from justllms.tools import tool

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        client, provider = _make_client({"enabled": True})
        assert client.cache is not None
        messages = [{"role": "user", "content": "add 1 and 2"}]

        # FakeProvider has no tool adapter, so the tools path raises -- but
        # crucially the cache is never consulted or written.
        with pytest.raises(Exception):
            client.completion.create(messages=messages, provider="fake", tools=[add])

        stats = client.cache.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0
