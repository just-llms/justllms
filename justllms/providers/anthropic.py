from typing import Any, Dict, List, Optional

from justllms.core.base import BaseProvider, BaseResponse
from justllms.core.models import Choice, Message, ModelInfo, Role, Usage
from justllms.tools.adapters.base import BaseToolAdapter


class AnthropicResponse(BaseResponse):
    """Anthropic-specific response implementation."""

    pass


class AnthropicProvider(BaseProvider):
    """Anthropic provider implementation."""

    supports_tools = True
    """Anthropic Claude supports tool use."""

    MODELS = {
        "claude-opus-4.1": ModelInfo(
            name="claude-opus-4.1",
            provider="anthropic",
            max_tokens=32000,
            max_context_length=200000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=15.0,
            cost_per_1k_completion_tokens=75.0,
            tags=["flagship", "most-capable", "multimodal", "extended-thinking"],
        ),
        "claude-sonnet-4": ModelInfo(
            name="claude-sonnet-4",
            provider="anthropic",
            max_tokens=64000,
            max_context_length=200000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=3.0,
            cost_per_1k_completion_tokens=15.0,
            tags=["high-performance", "multimodal", "extended-thinking"],
        ),
        "claude-haiku-3.5": ModelInfo(
            name="claude-haiku-3.5",
            provider="anthropic",
            max_tokens=8192,
            max_context_length=200000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.8,
            cost_per_1k_completion_tokens=4.0,
            tags=["fastest", "efficient", "multimodal"],
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
        url = f"{self.config.api_base or 'https://api.anthropic.com'}/v1/messages"

        system_message, formatted_messages = self._format_messages(messages)

        payload = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        if system_message:
            payload["system"] = system_message

        # Map common parameters
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            payload["stop_sequences"] = (
                kwargs["stop"] if isinstance(kwargs["stop"], list) else [kwargs["stop"]]
            )

        response_data = self._make_http_request(
            url=url,
            payload=payload,
            headers=self._get_headers(),
            timeout=timeout,
        )

        return self._parse_response(response_data, model)

    def get_tool_adapter(self) -> Optional[BaseToolAdapter]:
        """Return the Anthropic tool adapter."""
        from justllms.tools.adapters.anthropic import AnthropicToolAdapter

        return AnthropicToolAdapter()
