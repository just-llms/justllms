from typing import Any, Dict, List, Optional

from justllms.core.base import BaseResponse
from justllms.core.models import Message, ModelInfo
from justllms.core.openai_base import BaseOpenAIChatProvider
from justllms.tools.adapters.base import BaseToolAdapter


class GrokResponse(BaseResponse):
    """Grok-specific response implementation."""

    pass


class GrokProvider(BaseOpenAIChatProvider):
    """Grok (xAI) provider implementation.

    The xAI API is OpenAI-compatible, so this builds on
    :class:`BaseOpenAIChatProvider` to inherit streaming and tool calling.
    """

    supports_tools = True
    """xAI Grok exposes OpenAI-compatible function calling."""

    MODELS = {
        "grok-4": ModelInfo(
            name="grok-4",
            provider="grok",
            max_tokens=32768,
            max_context_length=130000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=6.0,
            cost_per_1k_completion_tokens=30.0,
            tags=["flagship", "most-intelligent", "multimodal", "coding", "latest"],
        ),
        "grok-4-heavy": ModelInfo(
            name="grok-4-heavy",
            provider="grok",
            max_tokens=32768,
            max_context_length=130000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=8.0,
            cost_per_1k_completion_tokens=40.0,
            tags=["heavy", "premium", "exclusive", "multimodal"],
        ),
        "grok-3": ModelInfo(
            name="grok-3",
            provider="grok",
            max_tokens=32768,
            max_context_length=131072,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=3.0,
            cost_per_1k_completion_tokens=15.0,
            tags=["advanced", "reasoning", "long-context"],
        ),
        "grok-3-speedy": ModelInfo(
            name="grok-3-speedy",
            provider="grok",
            max_tokens=32768,
            max_context_length=131072,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=5.0,
            cost_per_1k_completion_tokens=25.0,
            tags=["speedy", "premium", "fast"],
        ),
        "grok-3-mini": ModelInfo(
            name="grok-3-mini",
            provider="grok",
            max_tokens=16384,
            max_context_length=131072,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.3,
            cost_per_1k_completion_tokens=0.5,
            tags=["mini", "affordable", "efficient"],
        ),
        "grok-3-mini-speedy": ModelInfo(
            name="grok-3-mini-speedy",
            provider="grok",
            max_tokens=16384,
            max_context_length=131072,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.6,
            cost_per_1k_completion_tokens=4.0,
            tags=["mini", "speedy", "fast", "affordable"],
        ),
    }

    @property
    def name(self) -> str:
        return "grok"

    def get_available_models(self) -> Dict[str, ModelInfo]:
        return self.MODELS.copy()

    def _get_api_endpoint(self) -> str:
        """Get the xAI chat completions endpoint."""
        base_url = (self.config.api_base or "https://api.x.ai").rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return f"{base_url}/v1/chat/completions"

    def _get_request_headers(self) -> Dict[str, str]:
        """Generate HTTP headers for xAI API requests."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.config.headers)
        return headers

    def _format_messages_openai(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for xAI, converting unified image parts to image_url.

        The unified multimodal format uses ``{"type": "image", "image": {...}}``;
        xAI (like OpenAI) expects ``{"type": "image_url", "image_url": {...}}``.
        Tool/function fields are preserved by the base formatter.
        """
        formatted = self._format_messages_base(messages)

        for msg in formatted:
            content = msg.get("content")
            if isinstance(content, list):
                converted: List[Dict[str, Any]] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        converted.append({"type": "image_url", "image_url": item.get("image", {})})
                    else:
                        converted.append(item)
                msg["content"] = converted

        return formatted

    def get_tool_adapter(self) -> Optional[BaseToolAdapter]:
        """Return the OpenAI-compatible tool adapter (xAI shares the format)."""
        from justllms.tools.adapters.openai import OpenAIToolAdapter

        return OpenAIToolAdapter()
