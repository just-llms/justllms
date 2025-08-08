"""Main monitoring class that coordinates logging, metrics, and cost tracking."""

import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

from justllms.core.completion import CompletionResponse
from justllms.monitoring.cost_tracker import CostTracker
from justllms.monitoring.logger import JustLLMsLogger, LogLevel
from justllms.monitoring.metrics import MetricsCollector


class Monitor:
    """Main monitoring coordinator."""

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], Any]] = None,
        logger: Optional[JustLLMsLogger] = None,
        cost_tracker: Optional[CostTracker] = None,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        # Handle both dict and MonitoringConfig object
        if config is not None and hasattr(config, "model_dump"):
            # It's a Pydantic model, convert to dict
            self.config = config.model_dump()
        else:
            self.config = config or {}

        # Initialize components
        self.logger = logger or self._create_logger()
        self.cost_tracker = cost_tracker or self._create_cost_tracker()
        self.metrics = metrics_collector or self._create_metrics_collector()

        # Request tracking
        self.active_requests: Dict[str, Dict[str, Any]] = {}

    def _create_logger(self) -> JustLLMsLogger:
        """Create logger from config."""
        logger_config = self.config.get("logging", {})
        return JustLLMsLogger(
            name=logger_config.get("name", "justllms"),
            level=logger_config.get("level", LogLevel.INFO),
            console_output=logger_config.get("console_output", True),
            file_output=(
                Path(logger_config["file_output"]) if logger_config.get("file_output") else None
            ),
            rich_formatting=logger_config.get("rich_formatting", True),
        )

    def _create_cost_tracker(self) -> CostTracker:
        """Create cost tracker from config."""
        cost_config = self.config.get("cost_tracking", {})
        return CostTracker(
            persist_path=(
                Path(cost_config["persist_path"]) if cost_config.get("persist_path") else None
            ),
            budget_daily=cost_config.get("budget_daily"),
            budget_monthly=cost_config.get("budget_monthly"),
            budget_per_request=cost_config.get("budget_per_request"),
        )

    def _create_metrics_collector(self) -> MetricsCollector:
        """Create metrics collector from config."""
        metrics_config = self.config.get("metrics", {})
        return MetricsCollector(
            enable_opentelemetry=metrics_config.get("enable_opentelemetry", False),
            export_interval=metrics_config.get("export_interval", 60),
            custom_attributes=metrics_config.get("custom_attributes", {}),
        )

    def start_request(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        request_type: str = "completion",
        **metadata: Any,
    ) -> str:
        """Start tracking a request."""
        request_id = str(uuid.uuid4())

        self.active_requests[request_id] = {
            "start_time": time.time(),
            "provider": provider,
            "model": model,
            "request_type": request_type,
            "metadata": metadata,
        }

        if provider and model:
            self.logger.log_request(
                request_id=request_id,
                provider=provider,
                model=model,
                messages=metadata.get("messages", []),
                **metadata,
            )

            self.metrics.record_request(
                provider=provider,
                model=model,
                request_type=request_type,
                **metadata,
            )

        return request_id

    def end_request(
        self,
        request_id: str,
        response: CompletionResponse,
        **metadata: Any,
    ) -> None:
        """End tracking a request with a response."""
        if request_id not in self.active_requests:
            return

        request_data = self.active_requests.pop(request_id)
        duration_ms = (time.time() - request_data["start_time"]) * 1000

        provider = response.provider or request_data["provider"]
        model = response.model or request_data["model"]

        # Log the response
        self.logger.log_response(
            request_id=request_id,
            provider=provider,
            model=model,
            duration_ms=duration_ms,
            tokens_used=response.usage.total_tokens if response.usage else None,
            cost=response.usage.estimated_cost if response.usage else None,
            cached=response.cached,
            **metadata,
        )

        # Track metrics
        self.metrics.record_latency(
            provider=provider,
            model=model,
            latency_ms=duration_ms,
            **metadata,
        )

        if response.usage:
            self.metrics.record_tokens(
                provider=provider,
                model=model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                **metadata,
            )

            # Track costs
            result = self.cost_tracker.track_usage(
                provider=provider,
                model=model,
                usage=response.usage,
                request_id=request_id,
                metadata={**request_data["metadata"], **metadata},
            )

            # Log budget warnings
            for warning in result["warnings"]:
                self.logger.warning(warning, request_id=request_id)

    def end_streaming_request(
        self,
        request_id: str,
        **metadata: Any,
    ) -> None:
        """End tracking a streaming request."""
        if request_id not in self.active_requests:
            return

        request_data = self.active_requests.pop(request_id)
        duration_ms = (time.time() - request_data["start_time"]) * 1000

        provider = request_data["provider"]
        model = request_data["model"]

        if provider and model:
            self.logger.log_response(
                request_id=request_id,
                provider=provider,
                model=model,
                duration_ms=duration_ms,
                streaming=True,
                **metadata,
            )

            self.metrics.record_latency(
                provider=provider,
                model=model,
                latency_ms=duration_ms,
                streaming=True,
                **metadata,
            )

    def record_error(
        self,
        request_id: str,
        error: Exception,
        **metadata: Any,
    ) -> None:
        """Record an error for a request."""
        request_data = self.active_requests.pop(request_id, {})

        if request_data:
            duration_ms = (time.time() - request_data["start_time"]) * 1000
            provider = request_data["provider"]
            model = request_data["model"]

            if provider and model:
                self.logger.log_error_response(
                    request_id=request_id,
                    provider=provider,
                    model=model,
                    error=error,
                    duration_ms=duration_ms,
                    **metadata,
                )

                self.metrics.record_error(
                    provider=provider,
                    model=model,
                    error_type=type(error).__name__,
                    **metadata,
                )
        else:
            self.logger.error(
                f"Error for unknown request {request_id}: {error}",
                exception=error,
                **metadata,
            )

    def record_cache_hit(self, request_id: str, cache_key: Optional[str] = None) -> None:
        """Record a cache hit."""
        self.logger.log_cache_hit(request_id=request_id, cache_key=cache_key or "unknown")
        self.metrics.record_cache_hit(request_id=request_id)

    def record_cache_miss(self, request_id: str, cache_key: Optional[str] = None) -> None:
        """Record a cache miss."""
        self.logger.log_cache_miss(request_id=request_id, cache_key=cache_key or "unknown")
        self.metrics.record_cache_miss(request_id=request_id)

    def record_routing_decision(
        self,
        request_id: str,
        strategy: str,
        selected_provider: str,
        selected_model: str,
        reason: Optional[str] = None,
        **metadata: Any,
    ) -> None:
        """Record a routing decision."""
        self.logger.log_routing_decision(
            request_id=request_id,
            strategy=strategy,
            selected_provider=selected_provider,
            selected_model=selected_model,
            reason=reason,
            **metadata,
        )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all monitoring data."""
        return {
            "metrics": self.metrics.get_metrics_summary(),
            "cost_summary": {
                "daily": self.cost_tracker.get_cost_summary("daily"),
                "weekly": self.cost_tracker.get_cost_summary("weekly"),
                "monthly": self.cost_tracker.get_cost_summary("monthly"),
            },
            "active_requests": len(self.active_requests),
        }

    def set_log_level(self, level: Union[str, LogLevel]) -> None:
        """Set the log level."""
        self.logger.set_level(level)
