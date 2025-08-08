"""Unified analytics dashboard for cross-provider metrics."""

import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from justllms.analytics.models import (
    AnalyticsReport,
    CrossProviderMetrics,
    ModelStats,
    ProviderStats,
    ReportConfig,
    TimeSeriesData,
    UsageBreakdown,
)
from justllms.monitoring.cost_tracker import CostTracker
from justllms.monitoring.metrics import MetricsCollector


class AnalyticsDashboard:
    """Unified analytics dashboard for cross-provider LLM usage."""

    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.metrics_collector = metrics_collector
        self.cost_tracker = cost_tracker

        # In-memory aggregated data (for when no external collectors available)
        self._request_data: List[Dict[str, Any]] = []
        self._cost_data: List[Dict[str, Any]] = []
        self._error_data: List[Dict[str, Any]] = []
        self._latency_data: List[Dict[str, Any]] = []
        self._cache_data = {"hits": 0, "misses": 0}

    def add_request_data(
        self,
        provider: str,
        model: str,
        request_type: str = "completion",
        tokens_used: int = 0,
        cost: float = 0.0,
        latency_ms: float = 0.0,
        timestamp: Optional[datetime] = None,
        success: bool = True,
        cached: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add request data point to dashboard."""
        timestamp = timestamp or datetime.now()

        entry = {
            "timestamp": timestamp,
            "provider": provider,
            "model": model,
            "request_type": request_type,
            "tokens_used": tokens_used,
            "cost": cost,
            "latency_ms": latency_ms,
            "success": success,
            "cached": cached,
            "metadata": metadata or {},
        }

        self._request_data.append(entry)

        # Update cache stats
        if cached:
            self._cache_data["hits"] += 1
        else:
            self._cache_data["misses"] += 1

    def generate_report(self, config: Optional[ReportConfig] = None) -> AnalyticsReport:
        """Generate comprehensive analytics report."""
        config = config or ReportConfig()

        # Determine time period
        end_time = config.period_end or datetime.now()
        start_time = config.period_start or (end_time - timedelta(hours=24))

        # Filter data
        filtered_data = self._filter_data(start_time, end_time, config)

        # Generate cross-provider metrics
        cross_provider_metrics = self._calculate_cross_provider_metrics(filtered_data, config)

        # Generate usage breakdown
        usage_breakdown = self._calculate_usage_breakdown(filtered_data, config)

        # Generate time series data
        time_series = []
        if config.include_time_series:
            time_series = self._generate_time_series(filtered_data, start_time, end_time, config)

        # Generate top lists
        top_models_by_usage = self._get_top_models_by_usage(filtered_data, config.top_n)
        top_models_by_cost = self._get_top_models_by_cost(filtered_data, config.top_n)
        top_providers_by_requests = self._get_top_providers_by_requests(filtered_data, config.top_n)

        report = AnalyticsReport(
            report_id=str(uuid.uuid4()),
            period_start=start_time,
            period_end=end_time,
            cross_provider_metrics=cross_provider_metrics,
            usage_breakdown=usage_breakdown,
            time_series=time_series,
            top_models_by_usage=top_models_by_usage,
            top_models_by_cost=top_models_by_cost,
            top_providers_by_requests=top_providers_by_requests,
            filters={
                "providers": config.providers,
                "models": config.models,
                "custom_filters": config.custom_filters,
            },
        )

        return report

    def _filter_data(
        self, start_time: datetime, end_time: datetime, config: ReportConfig
    ) -> List[Dict[str, Any]]:
        """Filter request data based on time period and config."""
        filtered = []

        # Combine data from multiple sources
        data_sources = [self._request_data]

        # Add cost tracker data if available
        if self.cost_tracker:
            cost_history = self.cost_tracker.get_cost_history(limit=10000)
            for entry in cost_history:
                timestamp = datetime.fromisoformat(entry["timestamp"])
                if start_time <= timestamp <= end_time:
                    data_sources.append(
                        [
                            {
                                "timestamp": timestamp,
                                "provider": entry["provider"],
                                "model": entry["model"],
                                "request_type": "completion",
                                "tokens_used": entry["total_tokens"],
                                "cost": entry["estimated_cost"],
                                "latency_ms": 0.0,
                                "success": True,
                                "cached": False,
                                "metadata": entry.get("metadata", {}),
                            }
                        ]
                    )

        # Add metrics collector data if available
        if self.metrics_collector:
            # metrics_summary = self.metrics_collector.get_metrics_summary()
            # Convert metrics to standardized format
            # This is a simplified conversion - in practice you'd need more detailed data
            pass

        for source in data_sources:
            for entry in source:
                # Time filter
                if start_time <= entry["timestamp"] <= end_time:
                    # Provider filter
                    if config.providers and entry["provider"] not in config.providers:
                        continue

                    # Model filter
                    if config.models and entry["model"] not in config.models:
                        continue

                    filtered.append(entry)

        return filtered

    def _calculate_cross_provider_metrics(
        self, data: List[Dict[str, Any]], config: ReportConfig
    ) -> CrossProviderMetrics:
        """Calculate cross-provider summary metrics."""
        if not data:
            return CrossProviderMetrics()

        # Aggregate basic stats
        total_requests = len(data)
        total_tokens = sum(entry["tokens_used"] for entry in data)
        total_cost = sum(entry["cost"] for entry in data)
        total_errors = sum(1 for entry in data if not entry["success"])

        # Calculate unique counts
        unique_providers = len(set(entry["provider"] for entry in data))
        unique_models = len(set(f"{entry['provider']}:{entry['model']}" for entry in data))

        # Calculate cache hit rate
        cache_hits = self._cache_data["hits"]
        cache_misses = self._cache_data["misses"]
        total_cache_requests = cache_hits + cache_misses
        cache_hit_rate = (
            (cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0.0
        )

        # Calculate average latency
        latencies = [entry["latency_ms"] for entry in data if entry["latency_ms"] > 0]
        average_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

        # Calculate provider-level stats for rankings
        provider_stats = self._calculate_provider_stats(data)

        # Determine best providers
        most_used_provider = (
            max(provider_stats.keys(), key=lambda p: provider_stats[p]["requests"])
            if provider_stats
            else None
        )

        most_expensive_provider = (
            max(provider_stats.keys(), key=lambda p: provider_stats[p]["cost"])
            if provider_stats
            else None
        )

        fastest_provider = (
            min(provider_stats.keys(), key=lambda p: provider_stats[p]["avg_latency"])
            if provider_stats
            else None
        )

        most_reliable_provider = (
            max(provider_stats.keys(), key=lambda p: provider_stats[p]["success_rate"])
            if provider_stats
            else None
        )

        # Generate rankings
        cost_efficiency_ranking = sorted(
            provider_stats.keys(), key=lambda p: provider_stats[p]["cost_per_token"]
        )

        performance_ranking = sorted(
            provider_stats.keys(), key=lambda p: provider_stats[p]["avg_latency"]
        )

        reliability_ranking = sorted(
            provider_stats.keys(), key=lambda p: provider_stats[p]["success_rate"], reverse=True
        )

        return CrossProviderMetrics(
            total_requests=total_requests,
            total_tokens=total_tokens,
            total_cost=total_cost,
            total_errors=total_errors,
            unique_providers=unique_providers,
            unique_models=unique_models,
            cache_hit_rate=cache_hit_rate,
            average_latency_ms=average_latency_ms,
            most_used_provider=most_used_provider,
            most_expensive_provider=most_expensive_provider,
            fastest_provider=fastest_provider,
            most_reliable_provider=most_reliable_provider,
            cost_efficiency_ranking=cost_efficiency_ranking,
            performance_ranking=performance_ranking,
            reliability_ranking=reliability_ranking,
        )

    def _calculate_provider_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Calculate detailed stats for each provider."""
        stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "requests": 0,
                "tokens": 0,
                "cost": 0.0,
                "errors": 0,
                "latency_sum": 0.0,
                "latency_count": 0,
            }
        )

        for entry in data:
            provider = entry["provider"]
            stats[provider]["requests"] += 1
            stats[provider]["tokens"] += entry["tokens_used"]
            stats[provider]["cost"] += entry["cost"]

            if not entry["success"]:
                stats[provider]["errors"] += 1

            if entry["latency_ms"] > 0:
                stats[provider]["latency_sum"] += entry["latency_ms"]
                stats[provider]["latency_count"] += 1

        # Calculate derived metrics
        for _provider, provider_data in stats.items():
            provider_data["success_rate"] = (
                (
                    (provider_data["requests"] - provider_data["errors"])
                    / provider_data["requests"]
                    * 100
                )
                if provider_data["requests"] > 0
                else 0.0
            )
            provider_data["cost_per_token"] = (
                provider_data["cost"] / provider_data["tokens"]
                if provider_data["tokens"] > 0
                else 0.0
            )
            provider_data["avg_latency"] = (
                provider_data["latency_sum"] / provider_data["latency_count"]
                if provider_data["latency_count"] > 0
                else 0.0
            )

        return dict(stats)

    def _calculate_usage_breakdown(
        self, data: List[Dict[str, Any]], config: ReportConfig
    ) -> UsageBreakdown:
        """Calculate detailed usage breakdown."""
        provider_stats = {}
        model_stats = {}
        request_type_counts: Dict[str, int] = defaultdict(int)

        # Calculate provider stats
        provider_data = defaultdict(list)
        for entry in data:
            provider_data[entry["provider"]].append(entry)

        for provider, entries in provider_data.items():
            total_requests = len(entries)
            total_tokens = sum(e["tokens_used"] for e in entries)
            total_cost = sum(e["cost"] for e in entries)
            errors = sum(1 for e in entries if not e["success"])
            latencies = [e["latency_ms"] for e in entries if e["latency_ms"] > 0]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            models_used = list(set(e["model"] for e in entries))

            provider_stats[provider] = ProviderStats(
                provider=provider,
                total_requests=total_requests,
                total_tokens=total_tokens,
                total_cost=total_cost,
                error_count=errors,
                error_rate=(errors / total_requests * 100) if total_requests > 0 else 0.0,
                average_latency_ms=avg_latency,
                models_used=models_used,
            )

        # Calculate model stats
        model_data = defaultdict(list)
        for entry in data:
            key = f"{entry['provider']}:{entry['model']}"
            model_data[key].append(entry)

        for model_key, entries in model_data.items():
            provider, model = model_key.split(":", 1)
            total_requests = len(entries)
            total_tokens = sum(e["tokens_used"] for e in entries)
            total_cost = sum(e["cost"] for e in entries)
            errors = sum(1 for e in entries if not e["success"])
            latencies = [e["latency_ms"] for e in entries if e["latency_ms"] > 0]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

            model_stats[model_key] = ModelStats(
                provider=provider,
                model=model,
                total_requests=total_requests,
                total_tokens=total_tokens,
                total_cost=total_cost,
                error_count=errors,
                average_latency_ms=avg_latency,
            )

        # Calculate request type breakdown
        for entry in data:
            request_type_counts[entry["request_type"]] += 1

        # Generate time series (simplified - hourly for last 24 hours)
        hourly_data = self._generate_time_series(
            data, datetime.now() - timedelta(hours=24), datetime.now(), config
        )

        return UsageBreakdown(
            by_provider=provider_stats,
            by_model=model_stats,
            by_request_type=dict(request_type_counts),
            by_hour=hourly_data,
        )

    def _generate_time_series(
        self,
        data: List[Dict[str, Any]],
        start_time: datetime,
        end_time: datetime,
        config: ReportConfig,
    ) -> List[TimeSeriesData]:
        """Generate time series data."""
        if config.time_series_granularity == "hourly":
            interval = timedelta(hours=1)
        else:  # daily
            interval = timedelta(days=1)

        time_series = []
        current_time = start_time

        while current_time < end_time:
            next_time = current_time + interval

            # Filter data for this time window
            window_data = [
                entry for entry in data if current_time <= entry["timestamp"] < next_time
            ]

            requests = len(window_data)
            tokens = sum(entry["tokens_used"] for entry in window_data)
            cost = sum(entry["cost"] for entry in window_data)
            errors = sum(1 for entry in window_data if not entry["success"])
            latencies = [entry["latency_ms"] for entry in window_data if entry["latency_ms"] > 0]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

            time_series.append(
                TimeSeriesData(
                    timestamp=current_time,
                    requests=requests,
                    tokens=tokens,
                    cost=cost,
                    errors=errors,
                    latency_ms=avg_latency,
                )
            )

            current_time = next_time

        return time_series

    def _get_top_models_by_usage(self, data: List[Dict[str, Any]], top_n: int) -> List[ModelStats]:
        """Get top models by usage (request count)."""
        model_data = defaultdict(list)
        for entry in data:
            key = f"{entry['provider']}:{entry['model']}"
            model_data[key].append(entry)

        model_stats = []
        for model_key, entries in model_data.items():
            provider, model = model_key.split(":", 1)
            total_requests = len(entries)
            total_tokens = sum(e["tokens_used"] for e in entries)
            total_cost = sum(e["cost"] for e in entries)
            errors = sum(1 for e in entries if not e["success"])
            latencies = [e["latency_ms"] for e in entries if e["latency_ms"] > 0]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

            model_stats.append(
                ModelStats(
                    provider=provider,
                    model=model,
                    total_requests=total_requests,
                    total_tokens=total_tokens,
                    total_cost=total_cost,
                    error_count=errors,
                    average_latency_ms=avg_latency,
                )
            )

        return sorted(model_stats, key=lambda x: x.total_requests, reverse=True)[:top_n]

    def _get_top_models_by_cost(self, data: List[Dict[str, Any]], top_n: int) -> List[ModelStats]:
        """Get top models by cost."""
        model_stats = self._get_top_models_by_usage(data, len(data))  # Get all, then sort by cost
        return sorted(model_stats, key=lambda x: x.total_cost, reverse=True)[:top_n]

    def _get_top_providers_by_requests(
        self, data: List[Dict[str, Any]], top_n: int
    ) -> List[ProviderStats]:
        """Get top providers by request count."""
        provider_data = defaultdict(list)
        for entry in data:
            provider_data[entry["provider"]].append(entry)

        provider_stats = []
        for provider, entries in provider_data.items():
            total_requests = len(entries)
            total_tokens = sum(e["tokens_used"] for e in entries)
            total_cost = sum(e["cost"] for e in entries)
            errors = sum(1 for e in entries if not e["success"])
            latencies = [e["latency_ms"] for e in entries if e["latency_ms"] > 0]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            models_used = list(set(e["model"] for e in entries))

            provider_stats.append(
                ProviderStats(
                    provider=provider,
                    total_requests=total_requests,
                    total_tokens=total_tokens,
                    total_cost=total_cost,
                    error_count=errors,
                    error_rate=(errors / total_requests * 100) if total_requests > 0 else 0.0,
                    average_latency_ms=avg_latency,
                    models_used=models_used,
                )
            )

        return sorted(provider_stats, key=lambda x: x.total_requests, reverse=True)[:top_n]

    def get_live_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics summary."""
        recent_data = [
            entry
            for entry in self._request_data
            if entry["timestamp"] >= datetime.now() - timedelta(minutes=5)
        ]

        return {
            "timestamp": datetime.now().isoformat(),
            "recent_requests_5min": len(recent_data),
            "active_providers": len(set(entry["provider"] for entry in recent_data)),
            "cache_hit_rate": (
                self._cache_data["hits"]
                / (self._cache_data["hits"] + self._cache_data["misses"])
                * 100
                if (self._cache_data["hits"] + self._cache_data["misses"]) > 0
                else 0.0
            ),
            "total_requests_all_time": len(self._request_data),
        }
