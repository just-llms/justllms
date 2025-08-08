"""Health checking data models."""

import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthStatus(str, Enum):
    """Provider health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


class HealthCheckMethod(str, Enum):
    """Methods used for health checking."""

    LLM_REQUEST = "llm_request"
    ENDPOINT_PING = "endpoint_ping"
    STATUS_PAGE = "status_page"
    PASSIVE_MONITORING = "passive_monitoring"


class HealthResult(BaseModel):
    """Result of a health check."""

    provider: str = Field(..., description="Provider name")
    status: HealthStatus = Field(..., description="Overall health status")
    checked_at: datetime = Field(
        default_factory=datetime.now, description="When check was performed"
    )
    method: HealthCheckMethod = Field(..., description="Method used for checking")

    # Performance metrics
    latency_ms: Optional[float] = Field(
        default=None, description="Response latency in milliseconds"
    )
    success_rate: Optional[float] = Field(default=None, description="Success rate if applicable")

    # Technical details
    endpoint_reachable: bool = Field(default=True, description="Whether API endpoint is reachable")
    auth_valid: bool = Field(default=True, description="Whether authentication is valid")
    error_message: Optional[str] = Field(default=None, description="Error details if unhealthy")

    # Model and quota information
    models_available: List[str] = Field(default_factory=list, description="Available models")
    models_unavailable: List[str] = Field(default_factory=list, description="Unavailable models")
    rate_limit_remaining: Optional[int] = Field(default=None, description="Rate limit remaining")
    quota_usage: Optional[str] = Field(default=None, description="Quota usage information")

    # Cost tracking
    health_check_cost: Optional[float] = Field(
        default=None, description="Cost of this health check"
    )
    estimated_token_cost: Optional[float] = Field(
        default=None, description="Estimated cost per 1K tokens"
    )

    # Additional metadata
    raw_response: Optional[Dict[str, Any]] = Field(default=None, description="Raw API response")
    recommendations: List[str] = Field(default_factory=list, description="Health recommendations")

    @property
    def is_healthy(self) -> bool:
        """Check if provider is healthy."""
        return self.status == HealthStatus.HEALTHY

    @property
    def is_usable(self) -> bool:
        """Check if provider is usable (healthy or degraded)."""
        return self.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    @property
    def has_error(self) -> bool:
        """Check if there are any errors."""
        return self.error_message is not None

    def to_summary(self) -> Dict[str, Any]:
        """Get a summary of the health check result."""
        return {
            "provider": self.provider,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "error": self.error_message,
            "checked_at": self.checked_at.isoformat(),
        }


class HealthConfig(BaseModel):
    """Configuration for health checking."""

    # Request settings
    timeout_seconds: int = Field(default=10, description="Request timeout")
    max_retries: int = Field(default=2, description="Maximum retry attempts")

    # Cost control
    daily_budget_limit: float = Field(
        default=1.0, description="Daily budget limit for health checks"
    )
    confirm_before_spend: bool = Field(default=False, description="Ask user before spending money")
    warn_about_cost: bool = Field(default=True, description="Warn about estimated costs")

    # Caching
    cache_results: bool = Field(default=True, description="Cache health check results")
    cache_duration_seconds: int = Field(default=300, description="Cache duration (5 minutes)")

    # Test configuration
    test_models: Dict[str, str] = Field(
        default_factory=lambda: {
            "openai": "gpt-3.5-turbo",
            "google": "gemini-1.5-flash",
            "anthropic": "claude-3-haiku",
            "grok": "grok-2",
            "deepseek": "deepseek-v3",
        },
        description="Models to use for testing each provider",
    )

    max_tokens: int = Field(default=1, description="Maximum tokens for test requests")
    test_prompt: str = Field(default="Hi", description="Test prompt to send")

    # Detailed checking
    include_model_availability: bool = Field(default=True, description="Check model availability")
    include_rate_limit_info: bool = Field(
        default=True, description="Include rate limit information"
    )
    include_performance_metrics: bool = Field(
        default=True, description="Include performance metrics"
    )

    def get_test_config(self, provider: str) -> Dict[str, Any]:
        """Get test configuration for a specific provider."""
        base_config = {
            "timeout": self.timeout_seconds,
            "max_tokens": self.max_tokens,
        }

        # Provider-specific configurations
        if provider == "openai":
            return {
                **base_config,
                "model": self.test_models.get("openai", "gpt-3.5-turbo"),
                "messages": [{"role": "user", "content": self.test_prompt}],
                "temperature": 0,
            }
        elif provider == "google":
            return {
                **base_config,
                "model": self.test_models.get("google", "gemini-1.5-flash"),
                "prompt": self.test_prompt,
                "max_output_tokens": self.max_tokens,
                "temperature": 0,
            }
        elif provider == "anthropic":
            return {
                **base_config,
                "model": self.test_models.get("anthropic", "claude-3-haiku"),
                "messages": [{"role": "user", "content": self.test_prompt}],
                "max_tokens": self.max_tokens,
            }
        elif provider == "grok":
            return {
                **base_config,
                "model": self.test_models.get("grok", "grok-2"),
                "messages": [{"role": "user", "content": self.test_prompt}],
                "max_tokens": self.max_tokens,
                "temperature": 0,
            }
        elif provider == "deepseek":
            return {
                **base_config,
                "model": self.test_models.get("deepseek", "deepseek-v3"),
                "messages": [{"role": "user", "content": self.test_prompt}],
                "max_tokens": self.max_tokens,
                "temperature": 0,
            }
        else:
            return base_config


class HealthCheckCache:
    """Simple in-memory cache for health check results."""

    def __init__(self) -> None:
        self._cache: Dict[str, tuple[HealthResult, float]] = {}

    def get(self, key: str, max_age_seconds: int = 300) -> Optional[HealthResult]:
        """Get cached result if not expired."""
        if key not in self._cache:
            return None

        result, timestamp = self._cache[key]
        if time.time() - timestamp > max_age_seconds:
            del self._cache[key]
            return None

        return result

    def set(self, key: str, result: HealthResult) -> None:
        """Cache a health check result."""
        self._cache[key] = (result, time.time())

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_results": len(self._cache),
            "cache_keys": list(self._cache.keys()),
        }
