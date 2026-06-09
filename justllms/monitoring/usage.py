"""Usage tracking for completions across providers and models."""

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional

if TYPE_CHECKING:
    from justllms.core.models import Usage


@dataclass
class UsageRecord:
    """A single recorded completion's usage details."""

    timestamp: datetime
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    streamed: bool = False


def _empty_bucket() -> Dict[str, Any]:
    return {
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
    }


@dataclass
class _Totals:
    """Running totals kept separately from the (capped) records list."""

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    by_provider: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class UsageTracker:
    """Thread-safe accumulator of per-request usage and cost.

    Records every completion (token counts and estimated cost) and maintains
    running totals plus per-provider and per-model breakdowns. Individual
    records are kept in a ring buffer capped at ``max_records`` so
    long-running applications do not grow unbounded; running totals remain
    accurate even after old records are trimmed.
    """

    def __init__(self, max_records: int = 10000):
        """Initialize the tracker.

        Args:
            max_records: Maximum number of individual UsageRecord entries to
                retain. Older records are discarded first. Totals and
                breakdowns are unaffected by trimming.
        """
        self._lock = threading.Lock()
        self._records: Deque[UsageRecord] = deque(maxlen=max_records)
        self._totals = _Totals()

    def record(
        self,
        provider: str,
        model: str,
        usage: Optional["Usage"],
        streamed: bool = False,
    ) -> None:
        """Record usage for a completed request.

        Tolerates ``usage=None`` (the request is still counted, with zero
        tokens/cost) and missing ``estimated_cost`` (treated as 0.0).

        Args:
            provider: Provider name (e.g. "openai").
            model: Model identifier used for the request.
            usage: Usage object from the response, or None if unavailable.
            streamed: Whether the request was a streaming completion.
        """
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        cost = (usage.estimated_cost if usage and usage.estimated_cost else None) or 0.0

        entry = UsageRecord(
            timestamp=datetime.now(timezone.utc),
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            streamed=streamed,
        )

        with self._lock:
            self._records.append(entry)
            self._totals.requests += 1
            self._totals.prompt_tokens += prompt_tokens
            self._totals.completion_tokens += completion_tokens
            self._totals.total_tokens += total_tokens
            self._totals.cost += cost

            for bucket_map, key in (
                (self._totals.by_provider, provider),
                (self._totals.by_model, f"{provider}/{model}"),
            ):
                bucket = bucket_map.setdefault(key, _empty_bucket())
                bucket["requests"] += 1
                bucket["prompt_tokens"] += prompt_tokens
                bucket["completion_tokens"] += completion_tokens
                bucket["total_tokens"] += total_tokens
                bucket["cost"] += cost

    @property
    def total_cost(self) -> float:
        """Cumulative estimated cost (USD) across all recorded requests."""
        with self._lock:
            return self._totals.cost

    @property
    def total_tokens(self) -> int:
        """Cumulative total tokens across all recorded requests."""
        with self._lock:
            return self._totals.total_tokens

    @property
    def total_prompt_tokens(self) -> int:
        """Cumulative prompt tokens across all recorded requests."""
        with self._lock:
            return self._totals.prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        """Cumulative completion tokens across all recorded requests."""
        with self._lock:
            return self._totals.completion_tokens

    @property
    def request_count(self) -> int:
        """Number of recorded requests."""
        with self._lock:
            return self._totals.requests

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated usage totals with per-provider/per-model breakdowns.

        Returns:
            Dict with total_cost, total_tokens, total_prompt_tokens,
            total_completion_tokens, request_count, plus "by_provider" and
            "by_model" breakdowns (requests, tokens, cost each).
        """
        with self._lock:
            return {
                "total_cost": self._totals.cost,
                "total_tokens": self._totals.total_tokens,
                "total_prompt_tokens": self._totals.prompt_tokens,
                "total_completion_tokens": self._totals.completion_tokens,
                "request_count": self._totals.requests,
                "by_provider": {
                    name: dict(bucket) for name, bucket in self._totals.by_provider.items()
                },
                "by_model": {name: dict(bucket) for name, bucket in self._totals.by_model.items()},
            }

    def get_records(self) -> List[UsageRecord]:
        """Get a copy of the retained usage records (newest last)."""
        with self._lock:
            return list(self._records)

    def reset(self) -> None:
        """Clear all records and reset totals to zero."""
        with self._lock:
            self._records.clear()
            self._totals = _Totals()
