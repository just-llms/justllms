"""DeepSeek provider implementation."""

import time
from typing import Any, AsyncIterator, Dict, Iterator, List

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from justllms.core.base import BaseProvider, BaseResponse
from justllms.core.models import Choice, Message, ModelInfo, Role, Usage
from justllms.exceptions import ProviderError


class DeepSeekResponse(BaseResponse):
    """DeepSeek-specific response implementation."""

    pass


class DeepSeekProvider(BaseProvider):
    """DeepSeek provider implementation."""

    MODELS = {
        "deepseek-chat": ModelInfo(
            name="deepseek-chat",
            provider="deepseek",
            max_tokens=8192,
            max_context_length=65536,
            supports_functions=True,
            supports_vision=False,
            supports_streaming=True,
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
            supports_streaming=True,
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
            supports_streaming=True,
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
            supports_streaming=True,
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
        """Get the API endpoint."""
        base_url = self.config.api_base or "https://api.deepseek.com"
        return f"{base_url}/chat/completions"

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for DeepSeek API (OpenAI-compatible format)."""
        formatted_messages = []

        for msg in messages:
            formatted_msg = {"role": msg.role, "content": msg.content}

            # Handle function calls and tool use if needed
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                formatted_msg["tool_calls"] = msg.tool_calls

            formatted_messages.append(formatted_msg)

        return formatted_messages

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _parse_response(self, response_data: Dict[str, Any], model: str) -> DeepSeekResponse:
        """Parse DeepSeek API response."""
        choices_data = response_data.get("choices", [])

        if not choices_data:
            raise ProviderError("No choices in DeepSeek response")

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

        return DeepSeekResponse(
            id=response_data.get("id", f"deepseek-{int(time.time())}"),
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
        **kwargs: Any,
    ) -> BaseResponse:
        """Synchronous completion."""
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
                    "stream",
                    "tools",
                    "tool_choice",
                ]
                and v is not None
            },
        }

        with httpx.Client(timeout=self.config.timeout) as client:
            response = client.post(
                url,
                json=request_data,
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                raise ProviderError(f"DeepSeek API error: {response.status_code} - {response.text}")

            return self._parse_response(response.json(), model)

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
                    "stream",
                    "tools",
                    "tool_choice",
                ]
                and v is not None
            },
        }

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                url,
                json=request_data,
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                raise ProviderError(f"DeepSeek API error: {response.status_code} - {response.text}")

            return self._parse_response(response.json(), model)

    def stream(
        self,
        messages: List[Message],
        model: str,
        **kwargs: Any,
    ) -> Iterator[BaseResponse]:
        """Synchronous streaming completion."""
        url = self._get_api_endpoint()

        # Format request
        request_data = {
            "model": model,
            "messages": self._format_messages(messages),
            "stream": True,
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
                    "tools",
                    "tool_choice",
                ]
                and v is not None
            },
        }

        with httpx.Client(timeout=self.config.timeout) as client, client.stream(
            "POST",
            url,
            json=request_data,
            headers=self._get_headers(),
        ) as response:
            if response.status_code != 200:
                raise ProviderError(f"DeepSeek API error: {response.status_code}")

            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        import json

                        chunk_data = json.loads(data)

                        choices = chunk_data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                message = Message(role=Role.ASSISTANT, content=content)
                                choice = Choice(
                                    index=0,
                                    message=message,
                                    finish_reason=choices[0].get("finish_reason"),
                                )
                                yield DeepSeekResponse(
                                    id=chunk_data.get("id", f"deepseek-stream-{int(time.time())}"),
                                    model=model,
                                    choices=[choice],
                                    created=int(time.time()),
                                )
                    except json.JSONDecodeError:
                        continue

    async def astream(
        self,
        messages: List[Message],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[BaseResponse]:
        """Asynchronous streaming completion."""
        url = self._get_api_endpoint()

        # Format request
        request_data = {
            "model": model,
            "messages": self._format_messages(messages),
            "stream": True,
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
                    "tools",
                    "tool_choice",
                ]
                and v is not None
            },
        }

        async with httpx.AsyncClient(timeout=self.config.timeout) as client, client.stream(
            "POST",
            url,
            json=request_data,
            headers=self._get_headers(),
        ) as response:
            if response.status_code != 200:
                raise ProviderError(f"DeepSeek API error: {response.status_code}")

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        import json

                        chunk_data = json.loads(data)

                        choices = chunk_data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                message = Message(role=Role.ASSISTANT, content=content)
                                choice = Choice(
                                    index=0,
                                    message=message,
                                    finish_reason=choices[0].get("finish_reason"),
                                )
                                yield DeepSeekResponse(
                                    id=chunk_data.get("id", f"deepseek-stream-{int(time.time())}"),
                                    model=model,
                                    choices=[choice],
                                    created=int(time.time()),
                                )
                    except json.JSONDecodeError:
                        continue
