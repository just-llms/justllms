"""Failure-driven provider failover.

This module implements the failover layer that sits above per-request HTTP
retries (see ``BaseProvider._make_http_request``): when a provider ultimately
fails with a retryable category of error, the next (provider, model) pair in
the fallback chain is tried.
"""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import httpx

from justllms.exceptions import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
)
from justllms.exceptions import RequestTimeoutError

if TYPE_CHECKING:
    from justllms.core.base import BaseProvider
    from justllms.core.completion import CompletionResponse

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_ON = ["rate_limit", "server_error", "timeout", "connection", "auth"]


def classify_error(exc: Exception) -> Optional[str]:
    """Classify an exception into a failover category.

    Args:
        exc: The exception raised by a provider call.

    Returns:
        One of "rate_limit", "server_error", "timeout", "connection", "auth",
        or None when the error should NOT trigger failover (e.g. validation
        errors or 4xx client errors that would fail on every provider).
    """
    # Specific justllms exception subclasses first (they subclass ProviderError).
    if isinstance(exc, RateLimitError):
        return "rate_limit"
    if isinstance(exc, AuthenticationError):
        return "auth"
    if isinstance(exc, RequestTimeoutError):
        return "timeout"

    # httpx network errors. TimeoutException subclasses RequestError, so check it first.
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.RequestError):
        return "connection"

    if isinstance(exc, ProviderError):
        status = getattr(exc, "status_code", None)
        if not isinstance(status, int):
            return None
        if status == 429:
            return "rate_limit"
        if status == 408:
            return "timeout"
        if status in (401, 403):
            return "auth"
        if 500 <= status < 600:
            return "server_error"
        # Other 4xx (400 bad request, 404 model not found, ...) are caller
        # errors that would fail on every provider: do not fail over.
        return None

    # ValidationError, ValueError, ConfigurationError, anything unknown.
    return None


@dataclass
class FallbackAttempt:
    """Record of a single attempt in a fallback chain."""

    provider: str
    model: str
    error: Optional[str] = None
    error_category: Optional[str] = None
    succeeded: bool = False
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the attempt for response metadata."""
        return {
            "provider": self.provider,
            "model": self.model,
            "error": self.error,
            "error_category": self.error_category,
            "succeeded": self.succeeded,
            "duration_ms": self.duration_ms,
        }


def _read_setting(config: Any, key: str, default: Any) -> Any:
    """Read a setting from a dict-like or attribute-style config object."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class FallbackExecutor:
    """Builds and executes failure-driven fallback chains."""

    def __init__(self, routing_config: Any = None) -> None:
        """Initialize from a routing config (RoutingConfig, dict, or None).

        Args:
            routing_config: Object or mapping providing fallback_chain,
                auto_fallback, max_fallback_attempts and fallback_on.
        """
        self.fallback_chain: List[str] = list(
            _read_setting(routing_config, "fallback_chain", []) or []
        )
        self.auto_fallback: bool = bool(_read_setting(routing_config, "auto_fallback", False))
        self.max_fallback_attempts: int = int(
            _read_setting(routing_config, "max_fallback_attempts", 3)
        )
        self.fallback_on: List[str] = list(
            _read_setting(routing_config, "fallback_on", DEFAULT_FALLBACK_ON) or []
        )

    @property
    def enabled(self) -> bool:
        """Whether failover is active by configuration."""
        return bool(self.fallback_chain) or self.auto_fallback

    def _resolve_entry(
        self, entry: str, providers: Dict[str, "BaseProvider"]
    ) -> Optional[Tuple[str, str]]:
        """Resolve a chain entry ("provider/model" or "provider") to a tuple.

        Returns None (with a logged warning) for unknown providers, invalid
        models, or providers with no available models.
        """
        if "/" in entry:
            provider_name, model_name = entry.split("/", 1)
        else:
            provider_name, model_name = entry, ""

        provider = providers.get(provider_name)
        if provider is None:
            logger.warning(
                "Fallback chain entry '%s' skipped: provider '%s' is not available",
                entry,
                provider_name,
            )
            return None

        if model_name:
            if not provider.validate_model(model_name):
                logger.warning(
                    "Fallback chain entry '%s' skipped: model '%s' not found in provider '%s'",
                    entry,
                    model_name,
                    provider_name,
                )
                return None
            return provider_name, model_name

        models = provider.get_available_models()
        if not models:
            logger.warning(
                "Fallback chain entry '%s' skipped: provider '%s' has no available models",
                entry,
                provider_name,
            )
            return None
        return provider_name, next(iter(models.keys()))

    def build_chain(
        self,
        primary: Tuple[str, str],
        providers: Dict[str, "BaseProvider"],
    ) -> List[Tuple[str, str]]:
        """Build the ordered (provider, model) chain starting with the primary.

        Args:
            primary: The (provider, model) pair selected for the request.
            providers: Available provider instances keyed by name.

        Returns:
            Chain of unique (provider, model) pairs, primary first, truncated
            to max_fallback_attempts entries.
        """
        chain: List[Tuple[str, str]] = [primary]
        seen = {primary}

        for entry in self.fallback_chain:
            resolved = self._resolve_entry(entry, providers)
            if resolved is None or resolved in seen:
                continue
            chain.append(resolved)
            seen.add(resolved)

        if self.auto_fallback and not self.fallback_chain:
            for provider_name, provider in providers.items():
                if provider_name == primary[0]:
                    continue
                models = provider.get_available_models()
                if not models:
                    continue
                candidate = (provider_name, next(iter(models.keys())))
                if candidate in seen:
                    continue
                chain.append(candidate)
                seen.add(candidate)

        max_attempts = max(1, self.max_fallback_attempts)
        return chain[:max_attempts]

    def execute(
        self,
        chain: List[Tuple[str, str]],
        call: Callable[[str, str], "CompletionResponse"],
    ) -> Tuple["CompletionResponse", List[FallbackAttempt]]:
        """Try each (provider, model) pair in the chain until one succeeds.

        Args:
            chain: Ordered chain of (provider, model) pairs to attempt.
            call: Callable invoked as call(provider_name, model_name) that
                performs the completion and returns a CompletionResponse.

        Returns:
            Tuple of (successful response, list of all attempts including
            failed ones and the successful one).

        Raises:
            ValueError: If the chain is empty.
            Exception: The original error when it is non-retryable, not in
                fallback_on, or when the last attempt fails.
        """
        if not chain:
            raise ValueError("Fallback chain is empty")

        attempts: List[FallbackAttempt] = []

        for index, (provider_name, model_name) in enumerate(chain):
            start = time.monotonic()
            try:
                response = call(provider_name, model_name)
            except Exception as exc:
                duration_ms = (time.monotonic() - start) * 1000.0
                category = classify_error(exc)
                attempts.append(
                    FallbackAttempt(
                        provider=provider_name,
                        model=model_name,
                        error=str(exc),
                        error_category=category,
                        succeeded=False,
                        duration_ms=duration_ms,
                    )
                )

                is_last = index == len(chain) - 1
                if category is None or category not in self.fallback_on or is_last:
                    if is_last and len(attempts) > 1:
                        history = ", ".join(
                            f"{a.provider}/{a.model} ({a.error_category or 'unclassified'})"
                            for a in attempts
                        )
                        logger.warning(
                            "All %d fallback attempts failed: %s", len(attempts), history
                        )
                    raise

                next_provider, next_model = chain[index + 1]
                logger.warning(
                    "Provider %s failed (%s), falling back to %s/%s: %s",
                    provider_name,
                    category,
                    next_provider,
                    next_model,
                    exc,
                )
                continue

            duration_ms = (time.monotonic() - start) * 1000.0
            attempts.append(
                FallbackAttempt(
                    provider=provider_name,
                    model=model_name,
                    succeeded=True,
                    duration_ms=duration_ms,
                )
            )
            return response, attempts

        # Unreachable: the last failed attempt re-raises inside the loop.
        raise RuntimeError("Fallback chain exhausted without raising")  # pragma: no cover
