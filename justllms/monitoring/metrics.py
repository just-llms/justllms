"""Metrics collection for monitoring and observability."""

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Optional

try:
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )

    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False


class MetricsCollector:
    """Collect and export metrics for monitoring."""

    def __init__(
        self,
        enable_opentelemetry: bool = False,
        export_interval: int = 60,  # seconds
        custom_attributes: Optional[Dict[str, Any]] = None,
    ):
        self.enable_opentelemetry = enable_opentelemetry and HAS_OTEL
        self.export_interval = export_interval
        self.custom_attributes = custom_attributes or {}

        # In-memory metrics storage
        self.request_count: Dict[str, int] = defaultdict(int)
        self.error_count: Dict[str, int] = defaultdict(int)
        self.token_count: Dict[str, int] = defaultdict(int)
        self.latency_sum: Dict[str, float] = defaultdict(float)
        self.latency_count: Dict[str, int] = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0

        # OpenTelemetry setup
        if self.enable_opentelemetry:
            self._setup_opentelemetry()

    def _setup_opentelemetry(self) -> None:
        """Set up OpenTelemetry metrics."""
        reader = PeriodicExportingMetricReader(
            exporter=ConsoleMetricExporter(),
            export_interval_millis=self.export_interval * 1000,
        )
        provider = MeterProvider(metric_readers=[reader])
        metrics.set_meter_provider(provider)

        meter = metrics.get_meter("justllms")

        # Create instruments
        self.otel_request_counter = meter.create_counter(
            name="llm_requests_total",
            description="Total number of LLM requests",
            unit="requests",
        )

        self.otel_error_counter = meter.create_counter(
            name="llm_errors_total",
            description="Total number of LLM errors",
            unit="errors",
        )

        self.otel_token_counter = meter.create_counter(
            name="llm_tokens_total",
            description="Total number of tokens processed",
            unit="tokens",
        )

        self.otel_latency_histogram = meter.create_histogram(
            name="llm_request_duration",
            description="LLM request duration",
            unit="ms",
        )

        self.otel_cache_counter = meter.create_counter(
            name="llm_cache_requests",
            description="Cache hit/miss counter",
            unit="requests",
        )

    def record_request(
        self,
        provider: str,
        model: str,
        request_type: str = "completion",
        **attributes: Any,
    ) -> None:
        """Record a request."""
        key = f"{provider}:{model}:{request_type}"
        self.request_count[key] += 1

        if self.enable_opentelemetry:
            attrs = {
                "provider": provider,
                "model": model,
                "request_type": request_type,
                **self.custom_attributes,
                **attributes,
            }
            self.otel_request_counter.add(1, attrs)

    def record_error(
        self,
        provider: str,
        model: str,
        error_type: str,
        **attributes: Any,
    ) -> None:
        """Record an error."""
        key = f"{provider}:{model}:{error_type}"
        self.error_count[key] += 1

        if self.enable_opentelemetry:
            attrs = {
                "provider": provider,
                "model": model,
                "error_type": error_type,
                **self.custom_attributes,
                **attributes,
            }
            self.otel_error_counter.add(1, attrs)

    def record_tokens(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        **attributes: Any,
    ) -> None:
        """Record token usage."""
        key = f"{provider}:{model}"
        self.token_count[key] += prompt_tokens + completion_tokens

        if self.enable_opentelemetry:
            attrs = {
                "provider": provider,
                "model": model,
                **self.custom_attributes,
                **attributes,
            }
            self.otel_token_counter.add(prompt_tokens, {**attrs, "token_type": "prompt"})
            self.otel_token_counter.add(completion_tokens, {**attrs, "token_type": "completion"})

    def record_latency(
        self,
        provider: str,
        model: str,
        latency_ms: float,
        **attributes: Any,
    ) -> None:
        """Record request latency."""
        key = f"{provider}:{model}"
        self.latency_sum[key] += latency_ms
        self.latency_count[key] += 1

        if self.enable_opentelemetry:
            attrs = {
                "provider": provider,
                "model": model,
                **self.custom_attributes,
                **attributes,
            }
            self.otel_latency_histogram.record(latency_ms, attrs)

    def record_cache_hit(self, **attributes: Any) -> None:
        """Record a cache hit."""
        self.cache_hits += 1

        if self.enable_opentelemetry:
            attrs = {
                "cache_result": "hit",
                **self.custom_attributes,
                **attributes,
            }
            self.otel_cache_counter.add(1, attrs)

    def record_cache_miss(self, **attributes: Any) -> None:
        """Record a cache miss."""
        self.cache_misses += 1

        if self.enable_opentelemetry:
            attrs = {
                "cache_result": "miss",
                **self.custom_attributes,
                **attributes,
            }
            self.otel_cache_counter.add(1, attrs)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of collected metrics."""
        # Calculate average latencies
        avg_latencies = {}
        for key in self.latency_sum:
            if self.latency_count[key] > 0:
                avg_latencies[key] = self.latency_sum[key] / self.latency_count[key]

        # Calculate cache hit rate
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_cache_requests if total_cache_requests > 0 else 0.0

        return {
            "timestamp": datetime.now().isoformat(),
            "request_counts": dict(self.request_count),
            "error_counts": dict(self.error_count),
            "token_counts": dict(self.token_count),
            "average_latencies_ms": avg_latencies,
            "cache_stats": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": cache_hit_rate,
            },
        }

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.request_count.clear()
        self.error_count.clear()
        self.token_count.clear()
        self.latency_sum.clear()
        self.latency_count.clear()
        self.cache_hits = 0
        self.cache_misses = 0
