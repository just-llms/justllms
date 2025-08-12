"""OpenAI provider implementation."""

from typing import Any, AsyncIterator, Dict, Iterator, List

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from justllms.core.base import BaseProvider, BaseResponse
from justllms.core.models import Choice, Message, ModelInfo, Usage
from justllms.exceptions import ProviderError


class OpenAIResponse(BaseResponse):
    """OpenAI-specific response implementation."""

    pass


class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation."""

    MODELS = {
        "gpt-5": ModelInfo(
            name="gpt-5",
            provider="openai",
            max_tokens=128000,
            max_context_length=272000,
            supports_functions=True,
            supports_vision=True,
            supports_streaming=True,
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
            supports_streaming=True,
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
            supports_streaming=True,
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
            supports_streaming=True,
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
            supports_streaming=True,
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
            supports_streaming=True,
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
            supports_streaming=True,
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
            supports_streaming=True,
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
            supports_streaming=True,
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

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization

        headers.update(self.config.headers)
        return headers

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for OpenAI API."""
        formatted = []

        for msg in messages:
            formatted_msg: Dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content,
            }

            if msg.name:
                formatted_msg["name"] = msg.name
            if msg.function_call:
                formatted_msg["function_call"] = msg.function_call
            if msg.tool_calls:
                formatted_msg["tool_calls"] = msg.tool_calls

            formatted.append(formatted_msg)

        return formatted

    def _parse_response(self, response_data: Dict[str, Any]) -> OpenAIResponse:
        """Parse OpenAI API response."""
        choices = []

        for choice_data in response_data.get("choices", []):
            message_data = choice_data.get("message", {})
            message = Message(
                role=message_data.get("role", "assistant"),
                content=message_data.get("content", ""),
                name=message_data.get("name"),
                function_call=message_data.get("function_call"),
                tool_calls=message_data.get("tool_calls"),
            )

            choice = Choice(
                index=choice_data.get("index", 0),
                message=message,
                finish_reason=choice_data.get("finish_reason"),
                logprobs=choice_data.get("logprobs"),
            )
            choices.append(choice)

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
            if k not in ["id", "model", "choices", "usage", "created", "system_fingerprint"]
        }

        return OpenAIResponse(
            id=response_data.get("id", ""),
            model=response_data.get("model", ""),
            choices=choices,
            usage=usage,
            created=response_data.get("created"),
            system_fingerprint=response_data.get("system_fingerprint"),
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
        **kwargs: Any,
    ) -> BaseResponse:
        """Synchronous completion."""
        url = f"{self.config.api_base or 'https://api.openai.com'}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": self._format_messages(messages),
            **kwargs,
        }

        with httpx.Client(timeout=self.config.timeout) as client:
            response = client.post(
                url,
                json=payload,
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                raise ProviderError(f"OpenAI API error: {response.status_code} - {response.text}")

            return self._parse_response(response.json())

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def acomplete(
        self,
        messages: List[Message],
        model: str,
        **kwargs: Any,
    ) -> BaseResponse:
        """Asynchronous completion."""
        url = f"{self.config.api_base or 'https://api.openai.com'}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": self._format_messages(messages),
            **kwargs,
        }

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                url,
                json=payload,
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                raise ProviderError(f"OpenAI API error: {response.status_code} - {response.text}")

            return self._parse_response(response.json())

    def stream(
        self,
        messages: List[Message],
        model: str,
        **kwargs: Any,
    ) -> Iterator[BaseResponse]:
        """Synchronous streaming completion."""
        url = f"{self.config.api_base or 'https://api.openai.com'}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": self._format_messages(messages),
            "stream": True,
            **kwargs,
        }

        with httpx.Client(timeout=self.config.timeout) as client, client.stream(
            "POST",
            url,
            json=payload,
            headers=self._get_headers(),
        ) as response:
            if response.status_code != 200:
                raise ProviderError(f"OpenAI API error: {response.status_code}")

            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        import json

                        chunk = json.loads(data)
                        yield self._parse_response(chunk)
                    except json.JSONDecodeError:
                        continue

    async def astream(
        self,
        messages: List[Message],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[BaseResponse]:
        """Asynchronous streaming completion."""
        url = f"{self.config.api_base or 'https://api.openai.com'}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": self._format_messages(messages),
            "stream": True,
            **kwargs,
        }

        async with httpx.AsyncClient(timeout=self.config.timeout) as client, client.stream(
            "POST",
            url,
            json=payload,
            headers=self._get_headers(),
        ) as response:
            if response.status_code != 200:
                raise ProviderError(f"OpenAI API error: {response.status_code}")

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        import json

                        chunk = json.loads(data)
                        yield self._parse_response(chunk)
                    except json.JSONDecodeError:
                        continue
