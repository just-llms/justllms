import json
import logging
from typing import Any, Dict, Iterable, Iterator, List, Optional

import httpx

from justllms.core.base import BaseProvider, BaseResponse
from justllms.core.models import Choice, Message, ModelInfo, Role, Usage
from justllms.core.streaming import StreamChunk, SyncStreamResponse
from justllms.tools.adapters.base import BaseToolAdapter

logger = logging.getLogger(__name__)


class AnthropicResponse(BaseResponse):
    """Anthropic-specific response implementation."""

    pass


class AnthropicProvider(BaseProvider):
    """Anthropic provider implementation."""

    supports_tools = True
    """Anthropic Claude supports tool use."""

    MODELS = {
        "claude-opus-4-7": ModelInfo(
            name="claude-opus-4-7",
            provider="anthropic",
            max_tokens=128000,
            max_context_length=1000000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.005,
            cost_per_1k_completion_tokens=0.025,
            tags=["flagship", "latest", "agentic", "coding", "multimodal", "adaptive-thinking"],
        ),
        "claude-sonnet-4-6": ModelInfo(
            name="claude-sonnet-4-6",
            provider="anthropic",
            max_tokens=64000,
            max_context_length=1000000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.003,
            cost_per_1k_completion_tokens=0.015,
            tags=["balanced", "speed", "intelligence", "multimodal", "adaptive-thinking"],
        ),
        "claude-haiku-4-5": ModelInfo(
            name="claude-haiku-4-5",
            provider="anthropic",
            max_tokens=64000,
            max_context_length=200000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.001,
            cost_per_1k_completion_tokens=0.005,
            tags=["fastest", "efficient", "multimodal"],
        ),
        "claude-haiku-4-5-20251001": ModelInfo(
            name="claude-haiku-4-5-20251001",
            provider="anthropic",
            max_tokens=64000,
            max_context_length=200000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.001,
            cost_per_1k_completion_tokens=0.005,
            tags=["fastest", "efficient", "multimodal", "pinned"],
        ),
        "claude-opus-4-6": ModelInfo(
            name="claude-opus-4-6",
            provider="anthropic",
            max_tokens=128000,
            max_context_length=1000000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.005,
            cost_per_1k_completion_tokens=0.025,
            tags=["legacy", "flagship", "multimodal", "extended-thinking"],
        ),
        "claude-sonnet-4-5-20250929": ModelInfo(
            name="claude-sonnet-4-5-20250929",
            provider="anthropic",
            max_tokens=64000,
            max_context_length=200000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.003,
            cost_per_1k_completion_tokens=0.015,
            tags=["legacy", "high-performance", "multimodal", "extended-thinking"],
        ),
        "claude-opus-4-5-20251101": ModelInfo(
            name="claude-opus-4-5-20251101",
            provider="anthropic",
            max_tokens=64000,
            max_context_length=200000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.005,
            cost_per_1k_completion_tokens=0.025,
            tags=["legacy", "flagship", "multimodal", "extended-thinking"],
        ),
        "claude-opus-4-1-20250805": ModelInfo(
            name="claude-opus-4-1-20250805",
            provider="anthropic",
            max_tokens=32000,
            max_context_length=200000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.015,
            cost_per_1k_completion_tokens=0.075,
            tags=["legacy", "flagship", "multimodal", "extended-thinking"],
        ),
        "claude-sonnet-4-20250514": ModelInfo(
            name="claude-sonnet-4-20250514",
            provider="anthropic",
            max_tokens=64000,
            max_context_length=200000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.003,
            cost_per_1k_completion_tokens=0.015,
            tags=["deprecated", "legacy", "multimodal", "extended-thinking"],
        ),
        "claude-opus-4-20250514": ModelInfo(
            name="claude-opus-4-20250514",
            provider="anthropic",
            max_tokens=32000,
            max_context_length=200000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.015,
            cost_per_1k_completion_tokens=0.075,
            tags=["deprecated", "legacy", "multimodal", "extended-thinking"],
        ),
        "claude-opus-4.1": ModelInfo(
            name="claude-opus-4.1",
            provider="anthropic",
            max_tokens=32000,
            max_context_length=200000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.015,
            cost_per_1k_completion_tokens=0.075,
            tags=["legacy", "alias", "multimodal", "extended-thinking"],
        ),
        "claude-sonnet-4": ModelInfo(
            name="claude-sonnet-4",
            provider="anthropic",
            max_tokens=64000,
            max_context_length=200000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.003,
            cost_per_1k_completion_tokens=0.015,
            tags=["legacy", "alias", "multimodal", "extended-thinking"],
        ),
        "claude-haiku-3.5": ModelInfo(
            name="claude-haiku-3.5",
            provider="anthropic",
            max_tokens=8192,
            max_context_length=200000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.0008,
            cost_per_1k_completion_tokens=0.004,
            tags=["legacy", "fastest", "efficient", "multimodal"],
        ),
        "claude-3-5-sonnet-20241022": ModelInfo(
            name="claude-3-5-sonnet-20241022",
            provider="anthropic",
            max_tokens=8192,
            max_context_length=200000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.003,
            cost_per_1k_completion_tokens=0.015,
            tags=["legacy", "reasoning", "multimodal"],
        ),
        "claude-3-5-haiku-20241022": ModelInfo(
            name="claude-3-5-haiku-20241022",
            provider="anthropic",
            max_tokens=8192,
            max_context_length=200000,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.001,
            cost_per_1k_completion_tokens=0.005,
            tags=["legacy", "fast", "efficient"],
        ),
        "claude-3-opus-20240229": ModelInfo(
            name="claude-3-opus-20240229",
            provider="anthropic",
            max_tokens=4096,
            max_context_length=200000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.015,
            cost_per_1k_completion_tokens=0.075,
            tags=["legacy", "powerful", "reasoning"],
        ),
    }

    @property
    def name(self) -> str:
        return "anthropic"

    def get_available_models(self) -> Dict[str, ModelInfo]:
        return self.MODELS.copy()

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "x-api-key": self.config.api_key or "",
            "anthropic-version": self.config.api_version or "2023-06-01",
            "content-type": "application/json",
        }

        headers.update(self.config.headers)
        return headers

    def _format_messages(
        self, messages: List[Message]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Format messages for Anthropic API."""
        system_message = None
        formatted_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_message = msg.content if isinstance(msg.content, str) else str(msg.content)
            else:
                formatted_msg = {
                    "role": "user" if msg.role == Role.USER else "assistant",
                    "content": msg.content,
                }
                formatted_messages.append(formatted_msg)

        return system_message, formatted_messages

    def _parse_response(self, response_data: Dict[str, Any], model: str) -> AnthropicResponse:
        """Parse Anthropic API response."""
        content = response_data.get("content", [])

        text_content = ""
        for item in content:
            if item.get("type") == "text":
                text_content = item.get("text", "")
                break

        message = Message(
            role=Role.ASSISTANT,
            content=text_content,
        )

        choice = Choice(
            index=0,
            message=message,
            finish_reason=response_data.get("stop_reason"),
        )

        usage_data = response_data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
        )

        return self._create_base_response(  # type: ignore[return-value]
            AnthropicResponse,
            response_data,
            [choice],
            usage,
            model,
        )

    def _get_messages_endpoint(self) -> str:
        """Return the Anthropic Messages API endpoint URL."""
        base_url = (self.config.api_base or "https://api.anthropic.com").rstrip("/")
        return f"{base_url}/v1/messages"

    def _build_payload(self, messages: List[Message], model: str, **kwargs: Any) -> Dict[str, Any]:
        """Build the base Messages API payload shared by all request types.

        Args:
            messages: Conversation messages to format.
            model: Model identifier to use.
            **kwargs: Generation parameters (max_tokens, temperature, top_p,
                top_k, stop).

        Returns:
            A payload dict ready for the Messages API (without ``stream`` or
            ``tools``, which callers add as needed).
        """
        system_message, formatted_messages = self._format_messages(messages)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens") or 4096,
        }

        if system_message:
            payload["system"] = system_message

        # Map common parameters (Anthropic supports top_k natively).
        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]
        if kwargs.get("top_p") is not None:
            payload["top_p"] = kwargs["top_p"]
        if kwargs.get("top_k") is not None:
            payload["top_k"] = kwargs["top_k"]
        if kwargs.get("stop") is not None:
            payload["stop_sequences"] = (
                kwargs["stop"] if isinstance(kwargs["stop"], list) else [kwargs["stop"]]
            )

        return payload

    def complete(
        self,
        messages: List[Message],
        model: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> BaseResponse:
        """Synchronous completion.

        Args:
            messages: List of messages for the completion.
            model: Model identifier to use.
            timeout: Optional timeout in seconds. If None, no timeout is enforced.
            **kwargs: Additional provider-specific parameters.
        """
        payload = self._build_payload(messages, model, **kwargs)

        response_data = self._make_http_request(
            url=self._get_messages_endpoint(),
            payload=payload,
            headers=self._get_headers(),
            timeout=timeout,
        )

        return self._parse_response(response_data, model)

    def complete_with_tools(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: str = "",
        tool_choice: Optional[Any] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> BaseResponse:
        """Complete with tool use support.

        Unlike the default :class:`BaseProvider` implementation, the Messages
        API expects ``tools``/``tool_choice`` as top-level fields rather than
        OpenAI-style kwargs, so this override wires them in explicitly.

        Args:
            messages: List of messages for the completion.
            tools: Tool definitions already formatted by the Anthropic adapter.
            model: Model identifier to use.
            tool_choice: Tool selection config already formatted by the adapter.
            timeout: Optional timeout in seconds.
            **kwargs: Additional generation parameters.

        Returns:
            BaseResponse whose raw ``content`` array carries any ``tool_use``
            blocks for the adapter to extract.
        """
        payload = self._build_payload(messages, model, **kwargs)

        if tools:
            payload["tools"] = tools
            # tool_choice is only meaningful when tools are provided.
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        response_data = self._make_http_request(
            url=self._get_messages_endpoint(),
            payload=payload,
            headers=self._get_headers(),
            timeout=timeout,
        )

        return self._parse_response(response_data, model)

    def _iter_anthropic_chunks(self, lines: Iterable[str]) -> Iterator[StreamChunk]:
        """Convert Anthropic SSE ``data:`` lines into StreamChunks.

        Tracks ``input_tokens`` (delivered in ``message_start``) so the final
        chunk can report complete usage alongside ``output_tokens`` from the
        ``message_delta`` event.

        Args:
            lines: Iterable of raw SSE lines from the response body.

        Yields:
            StreamChunk objects for text deltas and a final usage/finish chunk.

        Raises:
            ProviderError: If the stream contains an error event.
        """
        input_tokens = 0

        for raw_line in lines:
            line = raw_line.strip()
            if not line or not line.startswith("data:"):
                continue

            data = line[len("data:") :].strip()
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                logger.warning("Failed to parse Anthropic SSE chunk: %s", data)
                continue

            event_type = event.get("type")

            if event_type == "message_start":
                usage = (event.get("message", {}) or {}).get("usage", {}) or {}
                input_tokens = usage.get("input_tokens", 0) or 0
            elif event_type == "content_block_delta":
                delta = event.get("delta", {}) or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yield StreamChunk(content=delta["text"], raw=event)
            elif event_type == "message_delta":
                delta = event.get("delta", {}) or {}
                usage_data = event.get("usage", {}) or {}
                output_tokens = usage_data.get("output_tokens", 0) or 0
                yield StreamChunk(
                    finish_reason=delta.get("stop_reason"),
                    usage=Usage(
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                    ),
                    raw=event,
                )
            elif event_type == "error":
                from justllms.exceptions import ProviderError

                error = event.get("error", {}) or {}
                raise ProviderError(
                    f"Anthropic streaming error: {error.get('message', 'unknown error')}",
                    provider=self.name,
                )

    def _stream_anthropic_response(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        timeout: Optional[float] = None,
    ) -> Iterator[StreamChunk]:
        """Open the streaming HTTP connection and yield parsed chunks.

        Args:
            url: Messages API endpoint URL.
            payload: Request payload (with ``stream`` enabled).
            headers: Request headers.
            timeout: Optional timeout in seconds.

        Yields:
            StreamChunk objects from the response.

        Raises:
            ProviderError: If the streaming request fails.
        """
        from justllms.exceptions import ProviderError

        try:
            with httpx.Client(timeout=timeout) as client, client.stream(
                "POST", url, json=payload, headers=headers
            ) as response:
                response.raise_for_status()
                yield from self._iter_anthropic_chunks(response.iter_lines())
        except (httpx.HTTPError, httpx.RequestError) as e:
            raise ProviderError(f"Anthropic streaming request failed: {str(e)}") from e

    def stream(
        self,
        messages: List[Message],
        model: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> SyncStreamResponse:
        """Stream a completion using the Messages API server-sent events.

        Args:
            messages: List of messages for the completion.
            model: Model identifier to use.
            timeout: Optional timeout in seconds.
            **kwargs: Additional generation parameters.

        Returns:
            SyncStreamResponse: Streaming response iterator.
        """
        payload = self._build_payload(messages, model, **kwargs)
        payload["stream"] = True

        stream_iter = self._stream_anthropic_response(
            url=self._get_messages_endpoint(),
            payload=payload,
            headers=self._get_headers(),
            timeout=timeout,
        )

        return SyncStreamResponse(
            provider=self, model=model, messages=messages, raw_stream=stream_iter
        )

    def supports_streaming(self) -> bool:
        """Anthropic Claude supports streaming."""
        return True

    def supports_streaming_for_model(self, model: str) -> bool:
        """Check if a model supports streaming (all Claude models do)."""
        return model in self.get_available_models()

    def get_tool_adapter(self) -> Optional[BaseToolAdapter]:
        """Return the Anthropic tool adapter."""
        from justllms.tools.adapters.anthropic import AnthropicToolAdapter

        return AnthropicToolAdapter()
