"""Response cache manager: key generation, serialization, and backend wrapper."""

import hashlib
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from justllms.cache.base import BaseCacheBackend
from justllms.core.models import Choice, Message, Usage

if TYPE_CHECKING:
    from justllms.core.completion import CompletionResponse

# Keys that control caching/transport behavior and must never affect cache keys.
_KEY_EXCLUDED_PARAMS = frozenset(
    {"cache", "cache_ttl", "stream", "tools", "tool_choice", "execute_tools", "timeout"}
)


def make_cache_key(
    provider: str,
    model: str,
    messages: List[Message],
    params: Dict[str, Any],
) -> str:
    """Build a deterministic cache key for a completion request.

    Canonicalizes provider, model, message contents, and the generation
    parameters that affect output into sorted compact JSON, then hashes
    with SHA-256.

    Args:
        provider: Provider name handling the request.
        model: Model identifier.
        messages: Conversation messages.
        params: Generation parameters (cache-control and transport keys
            are stripped; None values are ignored).

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    message_dicts = [msg.model_dump(mode="json", exclude_none=True) for msg in messages]
    effective_params = {
        key: value
        for key, value in params.items()
        if key not in _KEY_EXCLUDED_PARAMS and value is not None
    }
    payload = {
        "provider": provider,
        "model": model,
        "messages": message_dicts,
        "params": effective_params,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _response_to_cache_dict(response: "CompletionResponse") -> Dict[str, Any]:
    """Serialize a CompletionResponse to a JSON-friendly dictionary."""
    return {
        "id": response.id,
        "model": response.model,
        "choices": [choice.model_dump(mode="json") for choice in response.choices],
        "usage": response.usage.model_dump(mode="json") if response.usage else None,
        "created": response.created,
        "system_fingerprint": response.system_fingerprint,
        "provider": response.provider,
        "raw_response": response.raw_response,
    }


def _response_from_cache_dict(data: Dict[str, Any]) -> "CompletionResponse":
    """Reconstruct a CompletionResponse (with cached=True) from a stored dict."""
    from justllms.core.completion import CompletionResponse

    choices = [Choice(**choice_data) for choice_data in data.get("choices", [])]
    usage_data = data.get("usage")
    usage = Usage(**usage_data) if usage_data else None
    # Copy to avoid mutating dicts shared with the cache backend, and drop
    # any key that would collide with the explicit ``cached`` argument.
    raw_response = dict(data.get("raw_response") or {})
    raw_response.pop("cached", None)

    response = CompletionResponse(
        id=data.get("id", ""),
        model=data.get("model", ""),
        choices=choices,
        usage=usage,
        created=data.get("created"),
        system_fingerprint=data.get("system_fingerprint"),
        provider=data.get("provider"),
        cached=True,
        **raw_response,
    )
    return response


class ResponseCache:
    """High-level response cache combining a backend with key generation.

    Args:
        backend: Storage backend implementing BaseCacheBackend.
        default_ttl: Default time-to-live in seconds for stored entries.
            None disables expiry.
    """

    def __init__(self, backend: BaseCacheBackend, default_ttl: Optional[float] = 3600.0) -> None:
        self.backend = backend
        self.default_ttl = default_ttl

    def get(
        self,
        provider: str,
        model: str,
        messages: List[Message],
        params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Look up a cached response dict for the given request, if any."""
        key = make_cache_key(provider, model, messages, params)
        return self.backend.get(key)

    def set(
        self,
        provider: str,
        model: str,
        messages: List[Message],
        params: Dict[str, Any],
        value: Dict[str, Any],
        ttl: Optional[float] = None,
    ) -> None:
        """Store a serialized response dict for the given request.

        Args:
            provider: Provider name handling the request.
            model: Model identifier.
            messages: Conversation messages.
            params: Generation parameters used for the request.
            value: Serialized response dictionary to store.
            ttl: Per-entry TTL override in seconds; falls back to default_ttl.
        """
        key = make_cache_key(provider, model, messages, params)
        effective_ttl = ttl if ttl is not None else self.default_ttl
        self.backend.set(key, value, ttl=effective_ttl)

    def clear(self) -> None:
        """Remove all cached responses."""
        self.backend.clear()

    @property
    def stats(self) -> Dict[str, Any]:
        """Return backend statistics."""
        return self.backend.stats
