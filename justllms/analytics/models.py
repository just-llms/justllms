"""Analytics data models."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProviderStats(BaseModel):
    """Statistics for a single provider."""

    provider: str = Field(..., description="Provider name")
    total_requests: int = Field(default=0, description="Total requests made")
    total_tokens: int = Field(default=0, description="Total tokens used")
    total_cost: float = Field(default=0.0, description="Total estimated cost")
    error_count: int = Field(default=0, description="Number of errors")
    error_rate: float = Field(default=0.0, description="Error rate as percentage")
    average_latency_ms: float = Field(default=0.0, description="Average response latency")
    models_used: List[str] = Field(
        default_factory=list, description="Models used with this provider"
    )

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return ((self.total_requests - self.error_count) / self.total_requests) * 100

    @property
    def cost_per_token(self) -> float:
        """Calculate cost per token."""
        return self.total_cost / self.total_tokens if self.total_tokens > 0 else 0.0

    @property
    def cost_per_request(self) -> float:
        """Calculate cost per request."""
        return self.total_cost / self.total_requests if self.total_requests > 0 else 0.0


class ModelStats(BaseModel):
    """Statistics for a specific model."""

    provider: str = Field(..., description="Provider name")
    model: str = Field(..., description="Model name")
    total_requests: int = Field(default=0, description="Total requests made")
    total_tokens: int = Field(default=0, description="Total tokens used")
    total_cost: float = Field(default=0.0, description="Total estimated cost")
    error_count: int = Field(default=0, description="Number of errors")
    average_latency_ms: float = Field(default=0.0, description="Average response latency")

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return ((self.total_requests - self.error_count) / self.total_requests) * 100

    @property
    def cost_per_token(self) -> float:
        """Calculate cost per token."""
        return self.total_cost / self.total_tokens if self.total_tokens > 0 else 0.0


class TimeSeriesData(BaseModel):
    """Time series data point."""

    timestamp: datetime = Field(..., description="Data point timestamp")
    requests: int = Field(default=0, description="Number of requests")
    tokens: int = Field(default=0, description="Number of tokens")
    cost: float = Field(default=0.0, description="Cost amount")
    errors: int = Field(default=0, description="Number of errors")
    latency_ms: float = Field(default=0.0, description="Average latency")


class UsageBreakdown(BaseModel):
    """Usage breakdown by various dimensions."""

    by_provider: Dict[str, ProviderStats] = Field(default_factory=dict)
    by_model: Dict[str, ModelStats] = Field(default_factory=dict)
    by_request_type: Dict[str, int] = Field(
        default_factory=dict, description="Breakdown by request type"
    )
    by_hour: List[TimeSeriesData] = Field(default_factory=list, description="Hourly breakdown")
    by_day: List[TimeSeriesData] = Field(default_factory=list, description="Daily breakdown")


class CrossProviderMetrics(BaseModel):
    """Cross-provider comparison metrics."""

    total_requests: int = Field(default=0, description="Total requests across all providers")
    total_tokens: int = Field(default=0, description="Total tokens across all providers")
    total_cost: float = Field(default=0.0, description="Total cost across all providers")
    total_errors: int = Field(default=0, description="Total errors across all providers")
    unique_providers: int = Field(default=0, description="Number of unique providers used")
    unique_models: int = Field(default=0, description="Number of unique models used")

    cache_hit_rate: float = Field(default=0.0, description="Overall cache hit rate")
    average_latency_ms: float = Field(
        default=0.0, description="Average response latency across all providers"
    )

    most_used_provider: Optional[str] = Field(
        default=None, description="Most frequently used provider"
    )
    most_expensive_provider: Optional[str] = Field(
        default=None, description="Most expensive provider by total cost"
    )
    fastest_provider: Optional[str] = Field(default=None, description="Fastest provider by latency")
    most_reliable_provider: Optional[str] = Field(
        default=None, description="Most reliable provider by success rate"
    )

    cost_efficiency_ranking: List[str] = Field(
        default_factory=list, description="Providers ranked by cost efficiency"
    )
    performance_ranking: List[str] = Field(
        default_factory=list, description="Providers ranked by performance"
    )
    reliability_ranking: List[str] = Field(
        default_factory=list, description="Providers ranked by reliability"
    )

    @property
    def overall_success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_requests == 0:
            return 0.0
        return ((self.total_requests - self.total_errors) / self.total_requests) * 100

    @property
    def average_cost_per_request(self) -> float:
        """Calculate average cost per request."""
        return self.total_cost / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def average_cost_per_token(self) -> float:
        """Calculate average cost per token."""
        return self.total_cost / self.total_tokens if self.total_tokens > 0 else 0.0


class AnalyticsReport(BaseModel):
    """Complete analytics report."""

    report_id: str = Field(..., description="Unique report identifier")
    generated_at: datetime = Field(
        default_factory=datetime.now, description="Report generation timestamp"
    )
    period_start: datetime = Field(..., description="Report period start")
    period_end: datetime = Field(..., description="Report period end")

    # Summary metrics
    cross_provider_metrics: CrossProviderMetrics = Field(..., description="Cross-provider summary")
    usage_breakdown: UsageBreakdown = Field(..., description="Detailed usage breakdown")

    # Time series data
    time_series: List[TimeSeriesData] = Field(default_factory=list, description="Time series data")

    # Top lists
    top_models_by_usage: List[ModelStats] = Field(
        default_factory=list, description="Top models by usage"
    )
    top_models_by_cost: List[ModelStats] = Field(
        default_factory=list, description="Top models by cost"
    )
    top_providers_by_requests: List[ProviderStats] = Field(
        default_factory=list, description="Top providers by request count"
    )

    # Metadata
    filters: Dict[str, Any] = Field(default_factory=dict, description="Applied filters")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional report metadata")

    @property
    def period_duration_hours(self) -> float:
        """Calculate report period duration in hours."""
        delta = self.period_end - self.period_start
        return delta.total_seconds() / 3600

    @property
    def requests_per_hour(self) -> float:
        """Calculate requests per hour."""
        duration = self.period_duration_hours
        return self.cross_provider_metrics.total_requests / duration if duration > 0 else 0.0


class ReportConfig(BaseModel):
    """Configuration for report generation."""

    period_start: Optional[datetime] = Field(default=None, description="Report period start")
    period_end: Optional[datetime] = Field(default=None, description="Report period end")
    providers: Optional[List[str]] = Field(default=None, description="Filter by providers")
    models: Optional[List[str]] = Field(default=None, description="Filter by models")
    include_time_series: bool = Field(default=True, description="Include time series data")
    time_series_granularity: str = Field(
        default="hourly", description="Time series granularity: hourly, daily"
    )
    top_n: int = Field(default=10, description="Number of items to include in top lists")
    include_cache_stats: bool = Field(default=True, description="Include cache statistics")
    include_error_analysis: bool = Field(default=True, description="Include error analysis")
    custom_filters: Dict[str, Any] = Field(
        default_factory=dict, description="Custom filters to apply"
    )
