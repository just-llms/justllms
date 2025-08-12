"""Anthropic (Claude) provider implementation."""

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from justllms.core.base import BaseProvider, BaseResponse
from justllms.core.models import Choice, Message, ModelInfo, Role, Usage
from justllms.exceptions import ProviderError


class AnthropicResponse(BaseResponse):
    """Anthropic-specific response implementation."""

    pass


class AnthropicProvider(BaseProvider):
    """Anthropic provider implementation."""

    MODELS = {
        "claude-opus-4.1": ModelInfo(
            name="claude-opus-4.1",
            provider="anthropic",
            max_tokens=32000,
            max_context_length=200000,
            supports_functions=True,
            supports_vision=True,
            supports_streaming=True,
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
            supports_streaming=True,
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
            supports_streaming=True,
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
            supports_streaming=True,
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
            supports_streaming=True,
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
            supports_streaming=True,
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
            if msg.role == "system":
                system_message = msg.content if isinstance(msg.content, str) else str(msg.content)
            else:
                formatted_msg = {
                    "role": "user" if msg.role == "user" else "assistant",
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

        # Extract only the keys we want to avoid conflicts
        raw_response = {
            k: v for k, v in response_data.items() if k not in ["id", "model", "choices", "usage"]
        }

        return AnthropicResponse(
            id=response_data.get("id", ""),
            model=model,
            choices=[choice],
            usage=usage,
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

        with httpx.Client(timeout=self.config.timeout) as client:
            response = client.post(
                url,
                json=payload,
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                raise ProviderError(
                    f"Anthropic API error: {response.status_code} - {response.text}"
                )

            return self._parse_response(response.json(), model)

    async def acomplete(
        self,
        messages: List[Message],
        model: str,
        **kwargs: Any,
    ) -> BaseResponse:
        """Asynchronous completion."""
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

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                url,
                json=payload,
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                raise ProviderError(
                    f"Anthropic API error: {response.status_code} - {response.text}"
                )

            return self._parse_response(response.json(), model)

    def stream(  # noqa: C901
        self,
        messages: List[Message],
        model: str,
        **kwargs: Any,
    ) -> Iterator[BaseResponse]:
        """Synchronous streaming completion."""
        url = f"{self.config.api_base or 'https://api.anthropic.com'}/v1/messages"

        system_message, formatted_messages = self._format_messages(messages)

        payload = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "stream": True,
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

        with httpx.Client(timeout=self.config.timeout) as client, client.stream(
            "POST",
            url,
            json=payload,
            headers=self._get_headers(),
        ) as response:
            if response.status_code != 200:
                raise ProviderError(f"Anthropic API error: {response.status_code}")

            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        import json

                        chunk = json.loads(data)
                        if chunk.get("type") == "message_start":
                            continue
                        elif chunk.get("type") == "content_block_delta":
                            text = chunk.get("delta", {}).get("text", "")
                            if text:
                                message = Message(role=Role.ASSISTANT, content=text)
                                choice = Choice(index=0, message=message)
                                yield AnthropicResponse(
                                    id=chunk.get("id", ""),
                                    model=model,
                                    choices=[choice],
                                )
                    except json.JSONDecodeError:
                        continue

    async def astream(  # noqa: C901
        self,
        messages: List[Message],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[BaseResponse]:
        """Asynchronous streaming completion."""
        url = f"{self.config.api_base or 'https://api.anthropic.com'}/v1/messages"

        system_message, formatted_messages = self._format_messages(messages)

        payload = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "stream": True,
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

        async with httpx.AsyncClient(timeout=self.config.timeout) as client, client.stream(
            "POST",
            url,
            json=payload,
            headers=self._get_headers(),
        ) as response:
            if response.status_code != 200:
                raise ProviderError(f"Anthropic API error: {response.status_code}")

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        import json

                        chunk = json.loads(data)
                        if chunk.get("type") == "message_start":
                            continue
                        elif chunk.get("type") == "content_block_delta":
                            text = chunk.get("delta", {}).get("text", "")
                            if text:
                                message = Message(role=Role.ASSISTANT, content=text)
                                choice = Choice(index=0, message=message)
                                yield AnthropicResponse(
                                    id=chunk.get("id", ""),
                                    model=model,
                                    choices=[choice],
                                )
                    except json.JSONDecodeError:
                        continue
