from typing import Dict, Optional

from justllms.core.base import BaseResponse
from justllms.core.models import ModelInfo
from justllms.core.openai_base import BaseOpenAIChatProvider
from justllms.tools.adapters.base import BaseToolAdapter


class DeepSeekResponse(BaseResponse):
    pass


class DeepSeekProvider(BaseOpenAIChatProvider):
    """DeepSeek provider implementation."""

    supports_tools = True
    """DeepSeek exposes OpenAI-compatible function calling."""

    MODELS = {
        "deepseek-v4-flash": ModelInfo(
            name="deepseek-v4-flash",
            provider="deepseek",
            max_tokens=384000,
            max_context_length=1000000,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.00014,
            cost_per_1k_completion_tokens=0.00028,
            tags=["latest", "flash", "thinking", "json-output", "tool-calls", "long-context"],
        ),
        "deepseek-v4-flash-cached": ModelInfo(
            name="deepseek-v4-flash",
            provider="deepseek",
            max_tokens=384000,
            max_context_length=1000000,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.0000028,
            cost_per_1k_completion_tokens=0.00028,
            tags=["latest", "flash", "cached", "discount", "long-context"],
        ),
        "deepseek-v4-pro": ModelInfo(
            name="deepseek-v4-pro",
            provider="deepseek",
            max_tokens=384000,
            max_context_length=1000000,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.000435,
            cost_per_1k_completion_tokens=0.00087,
            tags=["latest", "pro", "thinking", "agentic", "coding", "long-context"],
        ),
        "deepseek-v4-pro-cached": ModelInfo(
            name="deepseek-v4-pro",
            provider="deepseek",
            max_tokens=384000,
            max_context_length=1000000,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.000003625,
            cost_per_1k_completion_tokens=0.00087,
            tags=["latest", "pro", "cached", "discount", "long-context"],
        ),
        "deepseek-chat": ModelInfo(
            name="deepseek-chat",
            provider="deepseek",
            max_tokens=384000,
            max_context_length=1000000,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.00014,
            cost_per_1k_completion_tokens=0.00028,
            tags=["legacy", "deprecated", "alias", "non-thinking"],
        ),
        "deepseek-chat-cached": ModelInfo(
            name="deepseek-chat",
            provider="deepseek",
            max_tokens=384000,
            max_context_length=1000000,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.0000028,
            cost_per_1k_completion_tokens=0.00028,
            tags=["legacy", "cached", "discount"],
        ),
        "deepseek-reasoner": ModelInfo(
            name="deepseek-reasoner",
            provider="deepseek",
            max_tokens=384000,
            max_context_length=1000000,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.00014,
            cost_per_1k_completion_tokens=0.00028,
            tags=["legacy", "deprecated", "alias", "thinking"],
        ),
        "deepseek-reasoner-cached": ModelInfo(
            name="deepseek-reasoner",
            provider="deepseek",
            max_tokens=384000,
            max_context_length=1000000,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.0000028,
            cost_per_1k_completion_tokens=0.00028,
            tags=["legacy", "cached", "discount", "thinking"],
        ),
    }

    @property
    def name(self) -> str:
        return "deepseek"

    def get_available_models(self) -> Dict[str, ModelInfo]:
        return self.MODELS.copy()

    def _get_api_endpoint(self) -> str:
        """Get DeepSeek chat completions endpoint."""
        base_url = self.config.api_base or "https://api.deepseek.com"
        return f"{base_url}/chat/completions"

    def _get_request_headers(self) -> Dict[str, str]:
        """Generate HTTP headers for DeepSeek API requests."""
        # Start with base headers if they exist
        headers = {}
        headers.update(
            {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
        )
        # Add any custom headers from config
        if self.config.headers:
            headers.update(self.config.headers)
        return headers

    def get_tool_adapter(self) -> Optional[BaseToolAdapter]:
        """Return the OpenAI-compatible tool adapter (DeepSeek shares the format)."""
        from justllms.tools.adapters.openai import OpenAIToolAdapter

        return OpenAIToolAdapter()
