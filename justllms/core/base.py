from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from justllms.core.models import Choice, Message, ModelInfo, ProviderConfig, Usage
from justllms.exceptions import ProviderError

if TYPE_CHECKING:
    from justllms.core.streaming import AsyncStreamResponse, SyncStreamResponse
    from justllms.tools.adapters.base import BaseToolAdapter

DEFAULT_TIMEOUT = 300.0


def _is_retryable_http_error(exc: BaseException) -> bool:
    """Determine if an exception is worth retrying.

    Retries on:
    - Network/connection errors (httpx.RequestError)
    - Rate limiting (429)
    - Server errors (500+)
    - Request timeout (408)

    Does NOT retry on:
    - Client errors (400-499 except 429, 408)
    """
    if isinstance(exc, httpx.RequestError):
        return True
    if isinstance(exc, ProviderError):
        status = getattr(exc, "status_code", None)
        if status is None:
            return False
        return status in (429, 408) or status >= 500
    return False


class BaseResponse:
    """Base class for all provider responses."""

    def __init__(
        self,
        id: str,
        model: str,
        choices: List[Choice],
        usage: Optional[Usage] = None,
        created: Optional[int] = None,
        system_fingerprint: Optional[str] = None,
        **kwargs: Any,
    ):
        self.id = id
        self.model = model
        self.choices = choices
        self.usage = usage
        self.created = created
        self.system_fingerprint = system_fingerprint
        self.raw_response = kwargs

    @property
    def content(self) -> Optional[str]:
        """Get the content of the first choice."""
        if self.choices and self.choices[0].message.content:
            content = self.choices[0].message.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list) and content:
                return str(content[0].get("text", ""))
        return None

    @property
    def message(self) -> Optional[Message]:
        """Get the message of the first choice."""
        if self.choices:
            return self.choices[0].message
        return None


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""

    requires_api_key: bool = True
    """Whether the provider needs an API key to initialize."""

    supports_tools: bool = False
    """Whether this provider supports tool/function calling."""

    supports_native_tools: bool = False
    """Whether this provider has native built-in tools."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._models_cache: Optional[Dict[str, ModelInfo]] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        pass

    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        model: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> BaseResponse:
        """Sync completion method.

        Args:
            messages: List of messages for the completion.
            model: Model identifier to use.
            timeout: Optional timeout in seconds. Defaults to 300 seconds if not specified.
            **kwargs: Additional provider-specific parameters.
        """
        pass

    @abstractmethod
    def get_available_models(self) -> Dict[str, ModelInfo]:
        """Get available models for this provider."""
        pass

    def validate_model(self, model: str) -> bool:
        """Validate if a model is available."""
        models = self.get_available_models()
        return model in models

    def get_model_info(self, model: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        models = self.get_available_models()
        return models.get(model)

    def estimate_cost(self, usage: Usage, model: str) -> Optional[float]:
        """Estimate the cost for the given usage."""
        model_info = self.get_model_info(model)
        if not model_info or not model_info.cost_per_1k_prompt_tokens:
            return None

        prompt_cost = (usage.prompt_tokens / 1000) * model_info.cost_per_1k_prompt_tokens
        completion_cost = (usage.completion_tokens / 1000) * (
            model_info.cost_per_1k_completion_tokens or 0
        )

        return prompt_cost + completion_cost

    def stream(
        self,
        messages: List[Message],
        model: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> "SyncStreamResponse | AsyncStreamResponse":
        """Stream completion - same parameters as complete().

        Default implementation raises NotImplementedError.
        Providers that support streaming should override this method.

        Args:
            messages: List of messages for the completion.
            model: Model identifier to use.
            timeout: Optional timeout in seconds. Defaults to 300 seconds if not specified.
            **kwargs: Additional provider-specific parameters.

        Returns:
            SyncStreamResponse or AsyncStreamResponse.

        Raises:
            NotImplementedError: If provider doesn't support streaming.
        """
        raise NotImplementedError(
            f"{self.name} provider doesn't support streaming. "
            f"Use complete() instead or switch to a streaming-capable provider."
        )

    def supports_streaming(self) -> bool:
        """Check if this provider supports streaming.

        Returns:
            True if streaming is supported, False otherwise.
        """
        return False

    def supports_streaming_for_model(self, model: str) -> bool:
        """Check if specific model supports streaming.

        Args:
            model: Model identifier to check.

        Returns:
            True if model supports streaming, False otherwise.
        """
        return False

    def count_tokens(self, text: Union[str, List[Dict[str, Any]]], model: str) -> int:
        """Count tokens in text or multimodal content.

        Uses tiktoken for accurate counting when available, falls back to
        character-based estimation. Providers can override for more accurate counting.

        Args:
            text: Text string or list of content items (for multimodal) to count tokens for.
            model: Model identifier for model-specific counting.

        Returns:
            Token count (accurate for OpenAI models, estimated for others).
        """
        from justllms.utils.token_counter import count_tokens

        # Handle multimodal content (list of dicts)
        if isinstance(text, list):
            # Extract text content from multimodal items
            total = 0
            for item in text:
                if isinstance(item, dict) and item.get("type") == "text":
                    total += count_tokens(item.get("text", ""), model)
                elif isinstance(item, dict) and item.get("type") == "image":
                    total += 85  # Rough estimate for images
            return total

        return count_tokens(text, model)

    def count_message_tokens(self, messages: List[Message], model: str) -> int:
        """Count tokens in messages.

        Uses tiktoken for accurate counting when available, including message
        overhead and formatting tokens. Falls back to character-based estimation.

        Args:
            messages: List of messages to count tokens for.
            model: Model identifier for model-specific counting.

        Returns:
            Token count (accurate for OpenAI models, estimated for others).
        """
        from justllms.utils.token_counter import count_tokens

        # Format messages to dict format expected by TokenCounter
        formatted = self._format_messages_base(messages)
        return count_tokens(formatted, model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception(_is_retryable_http_error),
        reraise=True,
    )
    def _make_http_request(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
        method: str = "POST",
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute HTTP request with automatic retry logic and error handling.

        Provides consistent HTTP request handling across all providers with
        built-in retry logic and standardized error reporting.

        Args:
            url: Target endpoint URL for the request.
            payload: Request body data to send as JSON.
            headers: Optional HTTP headers to include in request.
            params: Optional query parameters for the request.
            method: HTTP method to use ('POST' or 'GET').
            timeout: Optional timeout in seconds. Defaults to 300 seconds if not specified.

        Returns:
            Dict[str, Any]: Parsed JSON response from the API.

        Raises:
            ProviderError: If request fails after retries or returns non-200 status.
                          Includes provider name, status code, and response details.
            ValueError: If unsupported HTTP method is specified.
        """
        request_headers = headers or {}
        request_params = params or {}

        timeout_config = timeout if timeout is not None else DEFAULT_TIMEOUT
        with httpx.Client(timeout=timeout_config) as client:
            if method.upper() == "POST":
                response = client.post(
                    url,
                    json=payload,
                    headers=request_headers,
                    params=request_params,
                )
            elif method.upper() == "GET":
                response = client.get(
                    url,
                    headers=request_headers,
                    params=request_params,
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            if response.status_code != 200:
                raise ProviderError(
                    f"{self.name} API error: {response.status_code} - {response.text}",
                    provider=self.name,
                    status_code=response.status_code,
                    response_body=response.text,
                )

            return response.json()  # type: ignore[no-any-return]

    def _extract_raw_response(
        self, response_data: Dict[str, Any], exclude_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Filter response data to extract only non-standard fields.

        Removes standard response fields to prevent conflicts when creating
        response objects, while preserving provider-specific metadata.

        Args:
            response_data: Raw response dictionary from provider API.
            exclude_keys: Optional list of keys to exclude. Uses standard
                         keys if not provided.

        Returns:
            Dict[str, Any]: Filtered response data containing only custom fields.
        """
        default_exclude_keys = ["id", "model", "choices", "usage", "created", "system_fingerprint"]
        exclude_keys = default_exclude_keys if exclude_keys is None else exclude_keys

        return {k: v for k, v in response_data.items() if k not in exclude_keys}

    def _create_base_response(
        self,
        response_class: type,
        response_data: Dict[str, Any],
        choices: List[Choice],
        usage: Usage,
        model: str,
        **kwargs: Any,
    ) -> BaseResponse:
        """Construct a standardized response object from provider data.

        Creates response objects with consistent structure across all providers
        while preserving provider-specific metadata and allowing customization.

        Args:
            response_class: Response class to instantiate (e.g., OpenAIResponse).
            response_data: Raw response dictionary from provider API.
            choices: List of parsed choice objects with messages.
            usage: Token usage statistics for the request.
            model: Model identifier used for the request.
            **kwargs: Additional fields to include in the response object.

        Returns:
            BaseResponse: Instantiated response object with standard fields
                         and provider-specific metadata.
        """
        raw_response = self._extract_raw_response(response_data)

        return response_class(  # type: ignore[no-any-return]
            id=response_data.get("id", ""),
            model=model,
            choices=choices,
            usage=usage,
            created=response_data.get("created"),
            system_fingerprint=response_data.get("system_fingerprint"),
            **raw_response,
            **kwargs,
        )

    def _format_messages_base(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert Message objects to provider-compatible format.

        Provides standard message formatting that works with OpenAI-compatible APIs.
        Override this method in provider classes for custom message formatting.

        Args:
            messages: List of Message objects to format for API request.

        Returns:
            List[Dict[str, Any]]: List of formatted message dictionaries ready
                                 for API consumption.
        """
        formatted = []

        for msg in messages:
            formatted_msg: Dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content,
            }

            # Add optional fields if present
            if msg.name:
                formatted_msg["name"] = msg.name
            if msg.function_call:
                formatted_msg["function_call"] = msg.function_call
            if msg.tool_calls:
                formatted_msg["tool_calls"] = msg.tool_calls

            if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                formatted_msg["tool_call_id"] = msg.tool_call_id

            formatted.append(formatted_msg)

        return formatted

    def _create_standard_choice(self, message_data: Dict[str, Any], index: int = 0) -> Choice:
        """Create a standard Choice object from message data."""
        from justllms.core.models import Role

        message = Message(
            role=message_data.get("role", Role.ASSISTANT),
            content=message_data.get("content", ""),
            name=message_data.get("name"),
            function_call=message_data.get("function_call"),
            tool_calls=message_data.get("tool_calls"),
        )

        return Choice(
            index=index,
            message=message,
            finish_reason=message_data.get("finish_reason"),
            logprobs=message_data.get("logprobs"),
        )

    def _create_standard_usage(self, usage_data: Dict[str, Any]) -> Usage:
        """Create a standard Usage object from usage data."""
        return Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

    def _get_default_headers(self) -> Dict[str, str]:
        """Get common default headers that most providers use."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"justllms/{self.__class__.__name__}",
        }

        # Add custom headers from config
        headers.update(self.config.headers)
        return headers

    def complete_with_tools(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: str = "",
        tool_choice: Optional[Any] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> BaseResponse:
        """Complete with tool/function calling support.

        Default implementation delegates to complete() with tools in kwargs.
        Providers that support tools should override this method.

        Args:
            messages: List of messages for the completion.
            tools: List of tool definitions in provider format.
            model: Model identifier to use.
            tool_choice: Tool selection strategy.
            timeout: Optional timeout in seconds.
            **kwargs: Additional provider-specific parameters.

        Returns:
            BaseResponse with potential tool calls.
        """
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        return self.complete(messages, model, timeout=timeout, **kwargs)

    def get_tool_adapter(self) -> Optional["BaseToolAdapter"]:
        """Get the tool adapter for this provider.

        Returns:
            Tool adapter instance or None if tools not supported.

        Raises:
            NotImplementedError: If provider supports tools but hasn't
                                implemented this method.
        """
        if not self.supports_tools:
            return None

        raise NotImplementedError(
            f"{self.name} provider supports tools but hasn't implemented get_tool_adapter()"
        )
