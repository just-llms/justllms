import json
import logging
from typing import Any, Dict, Iterator, List, Optional

from justllms.core.base import BaseProvider, BaseResponse
from justllms.core.models import Message
from justllms.core.streaming import StreamChunk, SyncStreamResponse

logger = logging.getLogger(__name__)


class BaseOpenAIChatProvider(BaseProvider):
    """Base class for providers using OpenAI-compatible chat API.

    Provides common functionality for providers that follow the OpenAI chat
    completions API format, including standardized message formatting,
    request construction, and response parsing.

    Subclasses need to implement:
    - name: Provider identifier
    - get_available_models(): Provider's model catalog
    - _get_api_endpoint(): API endpoint URL
    - _get_request_headers(): Authentication headers
    - _customize_payload(): Provider-specific request modifications (optional)
    """

    def _format_messages_openai(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for OpenAI-compatible APIs.

        Converts Message objects to the standard OpenAI chat format with
        support for function calls, tool calls, and multimodal content.

        Args:
            messages: List of Message objects to format.

        Returns:
            List[Dict[str, Any]]: OpenAI-compatible message format.
        """
        return self._format_messages_base(messages)

    def _get_api_endpoint(self) -> str:
        """Get the chat completions endpoint URL.

        Must be implemented by subclasses to provide the correct API endpoint.

        Returns:
            str: Full URL for the chat completions endpoint.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement _get_api_endpoint")

    def _get_request_headers(self) -> Dict[str, str]:
        """Get authentication and request headers.

        Must be implemented by subclasses to provide provider-specific
        authentication headers.

        Returns:
            Dict[str, str]: Headers for API requests.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement _get_request_headers")

    def _customize_payload(self, payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Customize request payload for provider-specific requirements.

        Override this method to modify the request payload before sending.
        Default implementation returns payload unchanged.

        Args:
            payload: Base OpenAI-compatible request payload.
            **kwargs: Additional parameters from the complete() call.

        Returns:
            Dict[str, Any]: Modified payload for the API request.
        """
        return payload

    def _build_base_payload(
        self, messages: List[Message], model: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """Build base OpenAI-compatible payload with parameter filtering.

        Constructs the base payload with model and formatted messages, then filters
        and adds supported parameters while logging ignored/unknown parameters.

        Args:
            messages: Conversation messages to process.
            model: Model identifier for the request.
            **kwargs: Additional parameters to filter and add to payload.

        Returns:
            Dict[str, Any]: Base payload ready for API request.
        """
        # Build base payload
        payload = {
            "model": model,
            "messages": self._format_messages_openai(messages),
        }

        # Supported OpenAI parameters
        supported_params = {
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "n",
            "presence_penalty",
            "frequency_penalty",
            "tools",
            "tool_choice",
            "response_format",
            "seed",
            "user",
        }

        # Parameters to ignore (provider-specific or handled separately)
        ignored_params = {"top_k", "generation_config", "timeout"}

        # Filter and add parameters
        for key, value in kwargs.items():
            if value is not None:
                if key in ignored_params:
                    logger.debug(f"Parameter '{key}' is not supported by OpenAI. Ignoring.")
                elif key in supported_params:
                    payload[key] = value
                else:
                    logger.debug(f"Unknown parameter '{key}' ignored. Not in OpenAI API spec.")

        return payload

    def _parse_sse_line(self, line: str) -> Optional[StreamChunk]:
        """Parse a single SSE line into a StreamChunk.

        Args:
            line: Raw SSE line to parse.

        Returns:
            StreamChunk if line contains valid data, None otherwise.
        """
        line = line.strip()
        if not line:
            return None

        if not line.startswith("data: "):
            return None

        data = line[6:]  # Remove "data: " prefix

        if data == "[DONE]":
            return None  # Signal end of stream

        try:
            chunk_data = json.loads(data)
            choices = chunk_data.get("choices", [])

            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                finish_reason = choices[0].get("finish_reason")

                if content or finish_reason:
                    return StreamChunk(
                        content=content,
                        finish_reason=finish_reason,
                        raw=chunk_data,
                    )
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse SSE chunk: {data}")

        return None

    def _stream_sse_response(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        timeout: Optional[float] = None,
    ) -> Iterator[StreamChunk]:
        """Stream SSE response from OpenAI-compatible endpoint.

        Args:
            url: API endpoint URL.
            payload: Request payload (should have stream=True).
            headers: Request headers.
            timeout: Optional timeout in seconds.

        Yields:
            StreamChunk objects from the response.

        Raises:
            ProviderError: If the streaming request fails.
        """
        from justllms.core.streaming import parse_sse_stream

        return parse_sse_stream(
            url=url,
            payload=payload,
            headers=headers,
            parse_chunk_fn=self._parse_sse_line,
            timeout=timeout,
            error_prefix="Streaming request",
        )

    def _parse_openai_response(
        self, response_data: Dict[str, Any], model: str, response_class: type
    ) -> BaseResponse:
        """Parse OpenAI-compatible API response.

        Handles standard OpenAI response format with choices and usage data.
        Uses common parsing utilities from BaseProvider.

        Args:
            response_data: Raw JSON response from API.
            model: Model identifier used for the request.
            response_class: Response class to instantiate.

        Returns:
            BaseResponse: Parsed response object with choices and usage.
        """
        choices = []
        for choice_data in response_data.get("choices", []):
            message_data = choice_data.get("message", {})
            choice = self._create_standard_choice(
                {**message_data, "finish_reason": choice_data.get("finish_reason")},
                choice_data.get("index", 0),
            )
            choices.append(choice)

        usage = self._create_standard_usage(response_data.get("usage", {}))

        return self._create_base_response(
            response_class,
            response_data,
            choices,
            usage,
            model,
        )

    def complete(
        self,
        messages: List[Message],
        model: str,
        timeout: Any = None,
        **kwargs: Any,
    ) -> BaseResponse:
        """Execute OpenAI-compatible chat completion request.

        Constructs standardized request payload and handles the API call
        using common patterns for OpenAI-compatible providers.

        Args:
            messages: Conversation messages to process.
            model: Model identifier for the request.
            timeout: Optional timeout in seconds. If None, no timeout is enforced.
            **kwargs: Additional parameters (temperature, max_tokens, etc.).

        Returns:
            BaseResponse: Completed response from the provider.

        Raises:
            ProviderError: If the API request fails.
            NotImplementedError: If required methods are not implemented.
        """
        url = self._get_api_endpoint()

        # Build base payload with parameter filtering
        payload = self._build_base_payload(messages, model, **kwargs)

        # Allow provider-specific customization
        payload = self._customize_payload(payload, **kwargs)

        # Execute request using common HTTP handling
        response_data = self._make_http_request(
            url=url,
            payload=payload,
            headers=self._get_request_headers(),
            timeout=timeout,
        )
        try:
            import importlib

            module = importlib.import_module(self.__module__)
            response_class_name = f"{self.__class__.__name__.replace('Provider', 'Response')}"
            response_class = getattr(module, response_class_name, BaseResponse)
        except (ImportError, AttributeError):
            response_class = BaseResponse

        return self._parse_openai_response(response_data, model, response_class)

    def stream(
        self,
        messages: List[Message],
        model: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> SyncStreamResponse:
        """Stream completion using Server-Sent Events.

        Args:
            messages: Conversation messages to process.
            model: Model identifier for the request.
            timeout: Optional timeout in seconds.
            **kwargs: Additional parameters.

        Returns:
            SyncStreamResponse: Streaming response iterator.
        """
        url = self._get_api_endpoint()

        # Build base payload with parameter filtering
        payload = self._build_base_payload(messages, model, **kwargs)

        # Enable streaming
        payload["stream"] = True

        # Allow provider-specific customization
        payload = self._customize_payload(payload, **kwargs)

        # Use shared SSE streaming helper
        stream_iter = self._stream_sse_response(
            url=url,
            payload=payload,
            headers=self._get_request_headers(),
            timeout=timeout,
        )

        return SyncStreamResponse(
            provider=self, model=model, messages=messages, raw_stream=stream_iter
        )

    def supports_streaming(self) -> bool:
        """OpenAI-compatible providers support streaming."""
        return True

    def supports_streaming_for_model(self, model: str) -> bool:
        """Check if model supports streaming."""
        return model in self.get_available_models()
