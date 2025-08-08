"""Health checking implementation using dummy LLM calls."""

import asyncio
import hashlib
import time
from typing import Any, Dict, List, Optional

from justllms.exceptions import ProviderError
from justllms.health.models import (
    HealthCheckCache,
    HealthCheckMethod,
    HealthConfig,
    HealthResult,
    HealthStatus,
)


class EndpointHealthChecker:
    """Provider health checker using lightweight LLM requests."""

    def __init__(
        self,
        config: Optional[HealthConfig] = None,
        client: Optional[Any] = None,
    ):
        self.config = config or HealthConfig()
        self.client = client
        self.cache = HealthCheckCache()
        self._daily_spend = 0.0
        self._last_reset = time.time()

    async def check(self, provider: str, confirm_spend: Optional[bool] = None) -> HealthResult:
        """Perform health check on a provider using a lightweight LLM request."""
        # Check cache first
        if self.config.cache_results:
            cache_key = self._get_cache_key(provider)
            cached_result = self.cache.get(cache_key, self.config.cache_duration_seconds)
            if cached_result:
                return cached_result

        # Check budget
        estimated_cost = self._estimate_cost(provider)
        if not self._check_budget(estimated_cost, confirm_spend):
            return HealthResult(
                provider=provider,
                status=HealthStatus.UNKNOWN,
                method=HealthCheckMethod.LLM_REQUEST,
                error_message="Health check cancelled due to budget constraints",
            )

        # Perform actual health check
        result = await self._perform_health_check(provider)

        # Update spend tracking
        if result.health_check_cost:
            self._daily_spend += result.health_check_cost

        # Cache result
        if self.config.cache_results:
            cache_key = self._get_cache_key(provider)
            self.cache.set(cache_key, result)

        return result

    async def check_all(
        self, providers: Optional[List[str]] = None, confirm_spend: Optional[bool] = None
    ) -> Dict[str, HealthResult]:
        """Check health of multiple providers."""
        if providers is None:
            providers = ["openai", "google", "anthropic", "grok", "deepseek"]

        # Check total estimated cost
        total_estimated_cost = sum(self._estimate_cost(p) for p in providers)
        if not self._check_budget(total_estimated_cost, confirm_spend):
            return {
                provider: HealthResult(
                    provider=provider,
                    status=HealthStatus.UNKNOWN,
                    method=HealthCheckMethod.LLM_REQUEST,
                    error_message="Health check cancelled due to budget constraints",
                )
                for provider in providers
            }

        # Run checks concurrently
        tasks = [self.check(provider, confirm_spend=False) for provider in providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        health_results = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                health_results[providers[i]] = HealthResult(
                    provider=providers[i],
                    status=HealthStatus.DOWN,
                    method=HealthCheckMethod.LLM_REQUEST,
                    error_message=str(result),
                )
            else:
                health_results[providers[i]] = result  # type: ignore
        return health_results

    def estimate_cost(self, provider: str) -> float:
        """Estimate cost of health check for a provider."""
        return self._estimate_cost(provider)

    def get_daily_spend(self) -> float:
        """Get current daily spend on health checks."""
        self._reset_daily_spend_if_needed()
        return self._daily_spend

    def reset_daily_spend(self) -> None:
        """Reset daily spend counter."""
        self._daily_spend = 0.0
        self._last_reset = time.time()

    async def _perform_health_check(self, provider: str) -> HealthResult:
        """Perform the actual health check using a dummy LLM request."""
        if not self.client:
            return HealthResult(
                provider=provider,
                status=HealthStatus.UNKNOWN,
                method=HealthCheckMethod.LLM_REQUEST,
                error_message="No JustLLM client available for health checking",
            )

        test_config = self.config.get_test_config(provider)
        start_time = time.time()

        try:
            # Make a lightweight LLM request to test the provider
            response = await self._make_dummy_request(provider, test_config)
            latency_ms = (time.time() - start_time) * 1000

            # Parse response and create result
            return self._parse_response(provider, response, latency_ms, test_config)

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return self._handle_error(provider, e, latency_ms)

    async def _make_dummy_request(self, provider: str, config: Dict[str, Any]) -> Any:
        """Make a dummy LLM request to test the provider."""
        # Create minimal test messages based on provider
        if provider in ["openai", "anthropic", "grok", "deepseek"]:
            # These use messages format
            messages = [{"role": "user", "content": config.get("test_prompt", "Hi")}]
            if self.client:
                response = await self.client.completion.acreate(
                    provider=provider,
                    model=config.get("model"),
                    messages=messages,
                    max_tokens=config.get("max_tokens", 1),
                    temperature=config.get("temperature", 0),
                    timeout=config.get("timeout", 10),
                )
            else:
                raise ProviderError("Client not available")
        elif provider == "google":
            # Google uses different format
            messages = [{"role": "user", "content": config.get("test_prompt", "Hi")}]
            if self.client:
                response = await self.client.completion.acreate(
                    provider="google",
                    model=config.get("model"),
                    messages=messages,
                    max_tokens=config.get("max_tokens", 1),
                    temperature=config.get("temperature", 0),
                    timeout=config.get("timeout", 10),
                )
            else:
                raise ProviderError("Client not available")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        return response

    def _parse_response(  # noqa: C901
        self, provider: str, response: Any, latency_ms: float, test_config: Dict[str, Any]
    ) -> HealthResult:
        """Parse LLM response into health result."""
        try:
            # Extract basic info
            content = ""
            model = test_config.get("model", "unknown")

            # Handle response format
            if hasattr(response, "choices") and response.choices:
                content = response.choices[0].message.content if response.choices[0].message else ""

            if hasattr(response, "model"):
                model = response.model

            # Extract usage and cost info
            usage = getattr(response, "usage", None)
            cost = 0.0
            if usage and hasattr(usage, "estimated_cost"):
                cost = usage.estimated_cost
            else:
                # Use fallback cost estimate
                cost = self._estimate_cost(provider)

            # Determine health status
            has_response = bool(content and content.strip())

            if has_response and latency_ms < 30000:  # Less than 30 seconds
                status = HealthStatus.HEALTHY
            elif latency_ms >= 30000:
                status = HealthStatus.DEGRADED  # Slow but working
            else:
                status = HealthStatus.DEGRADED  # No response content

            # Generate recommendations
            recommendations = []
            if latency_ms < 2000:
                recommendations.append("Response time is excellent")
            elif latency_ms < 5000:
                recommendations.append("Response time is good")
            elif latency_ms < 10000:
                recommendations.append("Response time is acceptable")
            else:
                recommendations.append("Response time is slower than expected")

            if has_response:
                recommendations.append("Provider is responding normally")
            else:
                recommendations.append("Provider responded but with empty content")

            return HealthResult(
                provider=provider,
                status=status,
                method=HealthCheckMethod.LLM_REQUEST,
                latency_ms=latency_ms,
                success_rate=1.0 if has_response else 0.5,
                endpoint_reachable=True,
                auth_valid=True,
                models_available=[model] if model != "unknown" else [],
                health_check_cost=cost,
                estimated_token_cost=self._get_token_cost_estimate(provider),
                raw_response={
                    "content": content,
                    "model": model,
                    "usage": usage.model_dump() if usage and hasattr(usage, "model_dump") else None,
                },
                recommendations=recommendations,
            )

        except Exception as e:
            return HealthResult(
                provider=provider,
                status=HealthStatus.DEGRADED,
                method=HealthCheckMethod.LLM_REQUEST,
                latency_ms=latency_ms,
                endpoint_reachable=True,
                auth_valid=True,
                error_message=f"Failed to parse response: {str(e)}",
            )

    def _handle_error(self, provider: str, error: Exception, latency_ms: float) -> HealthResult:
        """Handle errors from LLM requests."""
        error_str = str(error).lower()

        # Determine status based on error type
        if any(
            term in error_str
            for term in ["unauthorized", "api key", "authentication", "forbidden", "401"]
        ):
            status = HealthStatus.DOWN
            auth_valid = False
            error_message = "Authentication failed - check API key"
        elif any(term in error_str for term in ["rate limit", "quota", "429"]):
            status = HealthStatus.DEGRADED
            auth_valid = True
            error_message = "Rate limited or quota exceeded"
        elif any(term in error_str for term in ["timeout", "connection", "network"]):
            status = HealthStatus.DOWN
            auth_valid = True
            error_message = "Connection timeout or network error"
        elif any(term in error_str for term in ["model", "not found", "unavailable", "404"]):
            status = HealthStatus.DEGRADED
            auth_valid = True
            error_message = "Requested model is not available"
        elif any(term in error_str for term in ["server error", "500", "502", "503"]):
            status = HealthStatus.DOWN
            auth_valid = True
            error_message = "Provider server error"
        else:
            status = HealthStatus.DOWN
            auth_valid = True
            error_message = f"Request failed: {str(error)}"

        return HealthResult(
            provider=provider,
            status=status,
            method=HealthCheckMethod.LLM_REQUEST,
            latency_ms=latency_ms,
            endpoint_reachable=not any(
                term in error_str for term in ["connection", "timeout", "network"]
            ),
            auth_valid=auth_valid,
            error_message=error_message,
            recommendations=self._get_error_recommendations(error_str),
        )

    def _get_error_recommendations(self, error_str: str) -> List[str]:
        """Get recommendations based on error type."""
        recommendations = []

        if any(term in error_str for term in ["api key", "unauthorized", "authentication"]):
            recommendations.extend(
                [
                    "Check that your API key is valid and properly configured",
                    "Verify API key has necessary permissions for this provider",
                ]
            )
        elif any(term in error_str for term in ["rate limit", "429"]):
            recommendations.extend(
                [
                    "You've hit rate limits - consider upgrading your plan",
                    "Implement request throttling in your application",
                ]
            )
        elif any(term in error_str for term in ["quota", "billing"]):
            recommendations.extend(
                [
                    "Check your account billing and usage limits",
                    "Verify your account is in good standing",
                ]
            )
        elif any(term in error_str for term in ["timeout", "connection", "network"]):
            recommendations.extend(
                [
                    "Check your internet connectivity",
                    "Provider may be experiencing temporary issues",
                    "Try again in a few minutes",
                ]
            )
        elif any(term in error_str for term in ["model", "not found", "unavailable"]):
            recommendations.extend(
                [
                    "Try a different model that's available in your region",
                    "Check provider documentation for model availability",
                ]
            )
        elif any(term in error_str for term in ["server error", "500", "502", "503"]):
            recommendations.extend(
                [
                    "Provider is experiencing server issues",
                    "Check provider status page for known outages",
                    "Try again later",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Check provider status page for known issues",
                    "Verify your account configuration and settings",
                ]
            )

        return recommendations

    def _estimate_cost(self, provider: str) -> float:
        """Estimate cost of health check for a provider."""
        # Conservative estimates for minimal test requests (1-2 tokens)
        cost_estimates = {
            "openai": 0.0001,  # ~1-2 tokens with gpt-3.5-turbo
            "google": 0.000001,  # Very cheap with Gemini Flash
            "anthropic": 0.00005,  # ~1-2 tokens with Claude Haiku
            "grok": 0.0,  # Free tier
            "deepseek": 0.000002,  # Very cheap
        }
        return cost_estimates.get(provider, 0.0001)

    def _get_token_cost_estimate(self, provider: str) -> float:
        """Get estimated cost per 1K tokens."""
        cost_per_1k = {
            "openai": 0.002,  # gpt-3.5-turbo
            "google": 0.001,  # Gemini Flash
            "anthropic": 0.25,  # Claude Haiku
            "grok": 0.0,  # Free
            "deepseek": 0.002,  # DeepSeek pricing
        }
        return cost_per_1k.get(provider, 0.002)

    def _check_budget(self, cost: float, confirm_spend: Optional[bool] = None) -> bool:
        """Check if we're within budget and get user confirmation if needed."""
        self._reset_daily_spend_if_needed()

        # Check daily budget
        if self._daily_spend + cost > self.config.daily_budget_limit:
            return False

        # Check if confirmation needed
        confirm = confirm_spend if confirm_spend is not None else self.config.confirm_before_spend
        if confirm and cost > 0:
            if self.config.warn_about_cost:
                # Health check cost warning would be shown here
                pass
            response = input("Continue? (y/N): ").lower().strip()
            return response in ["y", "yes"]

        return True

    def _reset_daily_spend_if_needed(self) -> None:
        """Reset daily spend counter if it's a new day."""
        current_time = time.time()
        # Reset if more than 24 hours have passed
        if current_time - self._last_reset > 86400:  # 24 hours in seconds
            self._daily_spend = 0.0
            self._last_reset = current_time

    def _get_cache_key(self, provider: str) -> str:
        """Generate cache key for provider health check."""
        config_str = f"{provider}:{self.config.test_models.get(provider)}:{self.config.max_tokens}"
        return hashlib.md5(config_str.encode()).hexdigest()
