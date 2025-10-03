from typing import Any, Dict, Optional


class JustLLMsError(Exception):
    """Base exception for all JustLLMs library errors.

    Provides common error handling patterns with structured error information
    including error codes and additional context details.

    Args:
        message: Human-readable error description.
        code: Optional error code for programmatic error handling.
        details: Optional dictionary with additional error context.
    """

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class ProviderError(JustLLMsError):
    """Error originating from an LLM provider API.

    Represents failures in communication with or responses from LLM provider
    APIs, including HTTP errors, API errors, and malformed responses.

    Args:
        message: Error description from provider or library.
        provider: Name of the provider that generated the error.
        status_code: HTTP status code if applicable.
        response_body: Raw response body from the provider API.
        **kwargs: Additional arguments passed to parent JustLLMsError.
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.provider = provider
        self.status_code = status_code
        self.response_body = response_body


class ValidationError(JustLLMsError):
    """Error during input validation."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.field = field
        self.value = value


class RateLimitError(ProviderError):
    """Rate limit exceeded error."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class TimeoutError(ProviderError):
    """Request timeout error."""

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.timeout_seconds = timeout_seconds


class AuthenticationError(ProviderError):
    """Authentication/authorization error."""

    def __init__(
        self,
        message: str,
        required_auth: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.required_auth = required_auth


class ConfigurationError(JustLLMsError):
    """Configuration error."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Any = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.config_value = config_value
