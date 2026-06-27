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

    # Pricing is per 1K tokens (USD). The xAI "grok-4"/"grok-3" generation was
    # retired on 2026-05-15 and now redirects to grok-4.3; the old slugs are
    # kept as legacy aliases for backward compatibility.
    MODELS = {
        "grok-4.3": ModelInfo(
            name="grok-4.3",
            provider="grok",
            max_tokens=32768,
            max_context_length=1000000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.00125,
            cost_per_1k_completion_tokens=0.0025,
            tags=["flagship", "latest", "most-intelligent", "multimodal", "coding"],
        ),
        "grok-4.20": ModelInfo(
            name="grok-4.20",
            provider="grok",
            max_tokens=32768,
            max_context_length=2000000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.00125,
            cost_per_1k_completion_tokens=0.0025,
            tags=["multimodal", "long-context", "reasoning"],
        ),
        "grok-4.1-fast": ModelInfo(
            name="grok-4.1-fast",
            provider="grok",
            max_tokens=32768,
            max_context_length=2000000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.0002,
            cost_per_1k_completion_tokens=0.0005,
            tags=["fast", "cost-efficient", "high-throughput", "long-context"],
        ),
        "grok-4": ModelInfo(
            name="grok-4",
            provider="grok",
            max_tokens=32768,
            max_context_length=1000000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.00125,
            cost_per_1k_completion_tokens=0.0025,
            tags=["legacy", "alias", "multimodal"],
        ),
        "grok-4-heavy": ModelInfo(
            name="grok-4-heavy",
            provider="grok",
            max_tokens=32768,
            max_context_length=1000000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.00125,
            cost_per_1k_completion_tokens=0.0025,
            tags=["legacy", "alias", "multimodal"],
        ),
        "grok-3": ModelInfo(
            name="grok-3",
            provider="grok",
            max_tokens=32768,
            max_context_length=131072,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.003,
            cost_per_1k_completion_tokens=0.015,
            tags=["legacy", "reasoning", "long-context"],
        ),
        "grok-3-mini": ModelInfo(
            name="grok-3-mini",
            provider="grok",
            max_tokens=16384,
            max_context_length=131072,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.0003,
            cost_per_1k_completion_tokens=0.0005,
            tags=["legacy", "mini", "affordable", "efficient"],
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
