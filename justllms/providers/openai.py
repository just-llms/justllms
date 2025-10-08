from typing import Dict, Optional

from justllms.core.base import BaseResponse
from justllms.core.models import ModelInfo
from justllms.core.openai_base import BaseOpenAIChatProvider
from justllms.tools.adapters.base import BaseToolAdapter


class OpenAIResponse(BaseResponse):
    """OpenAI-specific response implementation."""

    pass


class OpenAIProvider(BaseOpenAIChatProvider):
    """Simplified OpenAI provider implementation."""

    supports_tools = True
    """OpenAI supports function calling."""

    MODELS = {
        "gpt-5": ModelInfo(
            name="gpt-5",
            provider="openai",
            max_tokens=128000,
            max_context_length=272000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=1.25,
            cost_per_1k_completion_tokens=10.0,
            tags=["flagship", "reasoning", "multimodal", "long-context", "tool-chaining"],
        ),
        "gpt-5-mini": ModelInfo(
            name="gpt-5-mini",
            provider="openai",
            max_tokens=128000,
            max_context_length=272000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.3,
            cost_per_1k_completion_tokens=1.2,
            tags=["efficient", "multimodal", "long-context"],
        ),
        "gpt-4.1": ModelInfo(
            name="gpt-4.1",
            provider="openai",
            max_tokens=128000,
            max_context_length=1000000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.004,
            cost_per_1k_completion_tokens=0.012,
            tags=["reasoning", "multimodal", "long-context", "cost-efficient"],
        ),
        "gpt-4.1-nano": ModelInfo(
            name="gpt-4.1-nano",
            provider="openai",
            max_tokens=32000,
            max_context_length=128000,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.00008,
            cost_per_1k_completion_tokens=0.0003,
            tags=["fastest", "cheapest", "efficient"],
        ),
        "gpt-4o": ModelInfo(
            name="gpt-4o",
            provider="openai",
            max_tokens=16384,
            max_context_length=128000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.005,
            cost_per_1k_completion_tokens=0.015,
            tags=["multimodal", "general-purpose"],
        ),
        "gpt-4o-mini": ModelInfo(
            name="gpt-4o-mini",
            provider="openai",
            max_tokens=16384,
            max_context_length=128000,
            supports_functions=True,
            supports_vision=True,
            cost_per_1k_prompt_tokens=0.00015,
            cost_per_1k_completion_tokens=0.0006,
            tags=["multimodal", "efficient", "affordable"],
        ),
        "o1": ModelInfo(
            name="o1",
            provider="openai",
            max_tokens=100000,
            max_context_length=200000,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=15.0,
            cost_per_1k_completion_tokens=60.0,
            tags=["reasoning", "complex-tasks", "long-context"],
        ),
        "o4-mini": ModelInfo(
            name="o4-mini",
            provider="openai",
            max_tokens=100000,
            max_context_length=200000,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=3.0,
            cost_per_1k_completion_tokens=12.0,
            tags=["reasoning", "complex-tasks", "affordable"],
        ),
        "gpt-oss-120b": ModelInfo(
            name="gpt-oss-120b",
            provider="openai",
            max_tokens=32000,
            max_context_length=128000,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.0,
            cost_per_1k_completion_tokens=0.0,
            tags=["open-source", "code", "problem-solving", "tool-calling"],
        ),
    }

    @property
    def name(self) -> str:
        return "openai"

    def get_available_models(self) -> Dict[str, ModelInfo]:
        return self.MODELS.copy()

    def _get_api_endpoint(self) -> str:
        """Get OpenAI chat completions endpoint."""
        base_url = self.config.api_base or "https://api.openai.com"
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return f"{base_url}/v1/chat/completions"

    def _get_request_headers(self) -> Dict[str, str]:
        """Generate HTTP headers for OpenAI API requests."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization

        headers.update(self.config.headers)
        return headers

    def get_tool_adapter(self) -> Optional[BaseToolAdapter]:
        """Return the OpenAI tool adapter."""
        from justllms.tools.adapters.openai import OpenAIToolAdapter

        return OpenAIToolAdapter()
