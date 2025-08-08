"""Exception classes for JustLLMs."""

from typing import Any, Dict, Optional


class JustLLMsError(Exception):
    """Base exception for all JustLLMs errors."""

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
    """Error from an LLM provider."""

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


class RouteError(JustLLMsError):
    """Error during routing/model selection."""

    def __init__(
        self,
        message: str,
        strategy: Optional[str] = None,
        available_providers: Optional[list] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.strategy = strategy
        self.available_providers = available_providers


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
