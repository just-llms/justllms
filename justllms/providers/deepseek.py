from typing import Dict

from justllms.core.base import BaseResponse
from justllms.core.models import ModelInfo
from justllms.core.openai_base import BaseOpenAIChatProvider


class DeepSeekResponse(BaseResponse):
    pass


class DeepSeekProvider(BaseOpenAIChatProvider):
    """DeepSeek provider implementation."""

    MODELS = {
        "deepseek-chat": ModelInfo(
            name="deepseek-chat",
            provider="deepseek",
            max_tokens=8192,
            max_context_length=65536,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.27,
            cost_per_1k_completion_tokens=1.10,
            tags=["chat", "general-purpose", "json-output", "function-calling"],
        ),
        "deepseek-chat-cached": ModelInfo(
            name="deepseek-chat",
            provider="deepseek",
            max_tokens=8192,
            max_context_length=65536,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.07,
            cost_per_1k_completion_tokens=1.10,
            tags=["chat", "cached", "discount", "general-purpose"],
        ),
        "deepseek-reasoner": ModelInfo(
            name="deepseek-reasoner",
            provider="deepseek",
            max_tokens=65536,
            max_context_length=65536,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.55,
            cost_per_1k_completion_tokens=2.19,
            tags=["reasoning", "analysis", "complex-tasks", "json-output", "advanced"],
        ),
        "deepseek-reasoner-cached": ModelInfo(
            name="deepseek-reasoner",
            provider="deepseek",
            max_tokens=65536,
            max_context_length=65536,
            supports_functions=True,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.14,
            cost_per_1k_completion_tokens=2.19,
            tags=["reasoning", "cached", "discount", "advanced"],
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
