"""Logging functionality for JustLLMs."""

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

try:
    from rich.console import Console
    from rich.logging import RichHandler

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None  # type: ignore
    RichHandler = None  # type: ignore


class LogLevel(str, Enum):
    """Log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class JustLLMsLogger:
    """Custom logger for JustLLMs with rich formatting."""

    def __init__(
        self,
        name: str = "justllms",
        level: Union[str, LogLevel] = LogLevel.INFO,
        console_output: bool = True,
        file_output: Optional[Path] = None,
        rich_formatting: bool = True,
    ):
        self.name = name
        self.level = level.value if isinstance(level, LogLevel) else level
        self.console_output = console_output
        self.file_output = file_output
        self.rich_formatting = rich_formatting

        self.logger = self._setup_logger()
        self.console = Console() if rich_formatting and RICH_AVAILABLE else None

    def _setup_logger(self) -> logging.Logger:
        """Set up the logger with handlers."""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)

        # Remove existing handlers
        logger.handlers = []

        # Console handler
        if self.console_output:
            if self.rich_formatting and RICH_AVAILABLE:
                console_handler: logging.Handler = RichHandler(
                    rich_tracebacks=True,
                    tracebacks_show_locals=True,
                )
            else:
                console_handler = logging.StreamHandler(sys.stdout)
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                console_handler.setFormatter(formatter)

            console_handler.setLevel(self.level)
            logger.addHandler(console_handler)

        # File handler
        if self.file_output:
            self.file_output.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.file_output)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(self.level)
            logger.addHandler(file_handler)

        return logger

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs: Any) -> None:
        """Log error message."""
        if exception:
            self.logger.error(message, exc_info=exception, extra=kwargs)
        else:
            self.logger.error(message, extra=kwargs)

    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs: Any) -> None:
        """Log critical message."""
        if exception:
            self.logger.critical(message, exc_info=exception, extra=kwargs)
        else:
            self.logger.critical(message, extra=kwargs)

    def log_request(
        self,
        request_id: str,
        provider: str,
        model: str,
        messages: Any,
        **kwargs: Any,
    ) -> None:
        """Log an LLM request."""
        self.info(
            f"Request {request_id}: {provider}/{model}",
            request_id=request_id,
            provider=provider,
            model=model,
            message_count=len(messages) if hasattr(messages, "__len__") else 1,
            **kwargs,
        )

    def log_response(
        self,
        request_id: str,
        provider: str,
        model: str,
        duration_ms: float,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None,
        cached: bool = False,
        **kwargs: Any,
    ) -> None:
        """Log an LLM response."""
        message_parts = [
            f"Response {request_id}: {provider}/{model}",
            f"duration={duration_ms:.2f}ms",
        ]

        if tokens_used is not None:
            message_parts.append(f"tokens={tokens_used}")

        if cost is not None:
            message_parts.append(f"cost=${cost:.4f}")

        if cached:
            message_parts.append("(cached)")

        self.info(
            " ".join(message_parts),
            request_id=request_id,
            provider=provider,
            model=model,
            duration_ms=duration_ms,
            tokens_used=tokens_used,
            cost=cost,
            cached=cached,
            **kwargs,
        )

    def log_error_response(
        self,
        request_id: str,
        provider: str,
        model: str,
        error: Exception,
        duration_ms: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Log an error response."""
        message_parts = [
            f"Error {request_id}: {provider}/{model}",
            f"error={type(error).__name__}: {str(error)}",
        ]

        if duration_ms is not None:
            message_parts.append(f"duration={duration_ms:.2f}ms")

        self.error(
            " ".join(message_parts),
            exception=error,
            request_id=request_id,
            provider=provider,
            model=model,
            duration_ms=duration_ms,
            **kwargs,
        )

    def log_cache_hit(self, request_id: str, cache_key: str, **kwargs: Any) -> None:
        """Log a cache hit."""
        self.debug(
            f"Cache hit {request_id}: {cache_key[:32]}...",
            request_id=request_id,
            cache_key=cache_key,
            **kwargs,
        )

    def log_cache_miss(self, request_id: str, cache_key: str, **kwargs: Any) -> None:
        """Log a cache miss."""
        self.debug(
            f"Cache miss {request_id}: {cache_key[:32]}...",
            request_id=request_id,
            cache_key=cache_key,
            **kwargs,
        )

    def log_routing_decision(
        self,
        request_id: str,
        strategy: str,
        selected_provider: str,
        selected_model: str,
        reason: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a routing decision."""
        message_parts = [
            f"Routing {request_id}:",
            f"strategy={strategy}",
            f"selected={selected_provider}/{selected_model}",
        ]

        if reason:
            message_parts.append(f"reason={reason}")

        self.debug(
            " ".join(message_parts),
            request_id=request_id,
            strategy=strategy,
            selected_provider=selected_provider,
            selected_model=selected_model,
            reason=reason,
            **kwargs,
        )

    def set_level(self, level: Union[str, LogLevel]) -> None:
        """Set the log level."""
        level_value = level.value if isinstance(level, LogLevel) else level
        self.level = level_value
        self.logger.setLevel(level_value)
        for handler in self.logger.handlers:
            handler.setLevel(level_value)
