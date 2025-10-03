import time
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from justllms.core.base import DEFAULT_TIMEOUT, BaseProvider, BaseResponse
from justllms.core.models import Choice, Message, ModelInfo, Usage
from justllms.exceptions import ProviderError


class GrokResponse(BaseResponse):
    """Grok-specific response implementation."""

    pass


class GrokProvider(BaseProvider):
    """Grok provider implementation."""

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
        """Get the API endpoint."""
        base_url = self.config.api_base or "https://api.x.ai"
        return f"{base_url}/v1/chat/completions"

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for Grok API (OpenAI-compatible format)."""
        formatted_messages = []

        for msg in messages:
            formatted_msg: Dict[str, Any] = {"role": msg.role.value, "content": msg.content}

            # Handle multimodal content if supported
            if isinstance(msg.content, list):
                content_list: List[Dict[str, Any]] = []
                for item in msg.content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            content_list.append({"type": "text", "text": item.get("text", "")})
                        elif item.get("type") == "image":
                            content_list.append(
                                {"type": "image_url", "image_url": item.get("image", {})}
                            )
                formatted_msg["content"] = content_list

            formatted_messages.append(formatted_msg)

        return formatted_messages

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _parse_response(self, response_data: Dict[str, Any], model: str) -> GrokResponse:
        """Parse Grok API response."""
        choices_data = response_data.get("choices", [])

        if not choices_data:
            raise ProviderError("No choices in Grok response")

        # Parse choices
        choices = []
        for choice_data in choices_data:
            message_data = choice_data.get("message", {})
            message = Message(
                role=message_data.get("role", "assistant"),
                content=message_data.get("content", ""),
            )
            choice = Choice(
                index=choice_data.get("index", 0),
                message=message,
                finish_reason=choice_data.get("finish_reason", "stop"),
            )
            choices.append(choice)

        # Parse usage
        usage_data = response_data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        # Extract only the keys we want to avoid conflicts
        raw_response = {
            k: v
            for k, v in response_data.items()
            if k not in ["id", "model", "choices", "usage", "created"]
        }

        return GrokResponse(
            id=response_data.get("id", f"grok-{int(time.time())}"),
            model=model,
            choices=choices,
            usage=usage,
            created=response_data.get("created", int(time.time())),
            **raw_response,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
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
        url = self._get_api_endpoint()

        # Format request
        request_data = {
            "model": model,
            "messages": self._format_messages(messages),
            **{
                k: v
                for k, v in kwargs.items()
                if k
                in [
                    "temperature",
                    "max_tokens",
                    "top_p",
                    "frequency_penalty",
                    "presence_penalty",
                    "stop",
                ]
                and v is not None
            },
        }

        timeout_config = timeout if timeout is not None else DEFAULT_TIMEOUT

        with httpx.Client(timeout=timeout_config) as client:
            response = client.post(
                url,
                json=request_data,
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                raise ProviderError(f"Grok API error: {response.status_code} - {response.text}")

            return self._parse_response(response.json(), model)
