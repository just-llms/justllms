"""Azure OpenAI provider implementation."""

from typing import Any, AsyncIterator, Dict, Iterator, List

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from justllms.core.base import BaseProvider, BaseResponse
from justllms.core.models import Choice, Message, ModelInfo, Usage
from justllms.exceptions import ProviderError


class AzureOpenAIResponse(BaseResponse):
    """Azure OpenAI-specific response implementation."""

    pass


class AzureOpenAIProvider(BaseProvider):
    """Azure OpenAI provider implementation."""

    # Azure OpenAI models with deployment name mapping
    MODELS = {
        "gpt-5": ModelInfo(
            name="gpt-5",
            provider="azure_openai",
            max_tokens=128000,
            max_context_length=272000,
            supports_functions=True,
            supports_vision=True,
            supports_streaming=True,
            cost_per_1k_prompt_tokens=1.25,
            cost_per_1k_completion_tokens=10.0,
            tags=["flagship", "reasoning", "multimodal", "long-context"],
        ),
        "gpt-5-mini": ModelInfo(
            name="gpt-5-mini",
            provider="azure_openai",
            max_tokens=128000,
            max_context_length=272000,
            supports_functions=True,
            supports_vision=True,
            supports_streaming=True,
            cost_per_1k_prompt_tokens=0.3,
            cost_per_1k_completion_tokens=1.2,
            tags=["efficient", "multimodal", "long-context"],
        ),
        "gpt-5-nano": ModelInfo(
            name="gpt-5-nano",
            provider="azure_openai",
            max_tokens=128000,
            max_context_length=272000,
            supports_functions=True,
            supports_vision=True,
            supports_streaming=True,
            cost_per_1k_prompt_tokens=0.15,
            cost_per_1k_completion_tokens=0.6,
            tags=["nano", "affordable", "multimodal", "long-context"],
        ),
        "gpt-5-chat": ModelInfo(
            name="gpt-5-chat",
            provider="azure_openai",
            max_tokens=16384,
            max_context_length=128000,
            supports_functions=True,
            supports_vision=True,
            supports_streaming=True,
            cost_per_1k_prompt_tokens=0.8,
            cost_per_1k_completion_tokens=3.2,
            tags=["chat", "multimodal", "conversational"],
        ),
        "gpt-4o": ModelInfo(
            name="gpt-4o",
            provider="azure_openai",
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
            provider="azure_openai",
            max_tokens=16384,
            max_context_length=128000,
            supports_functions=True,
            supports_vision=True,
            supports_streaming=True,
            cost_per_1k_prompt_tokens=0.00015,
            cost_per_1k_completion_tokens=0.0006,
            tags=["multimodal", "efficient", "affordable"],
        ),
        "o4-mini": ModelInfo(
            name="o4-mini",
            provider="azure_openai",
            max_tokens=100000,
            max_context_length=200000,
            supports_functions=True,
            supports_vision=False,
            supports_streaming=True,
            cost_per_1k_prompt_tokens=3.0,
            cost_per_1k_completion_tokens=12.0,
            tags=["reasoning", "complex-tasks", "long-context"],
        ),
        "o3": ModelInfo(
            name="o3",
            provider="azure_openai",
            max_tokens=100000,
            max_context_length=200000,
            supports_functions=True,
            supports_vision=False,
            supports_streaming=True,
            cost_per_1k_prompt_tokens=15.0,
            cost_per_1k_completion_tokens=60.0,
            tags=["reasoning", "advanced", "complex-tasks"],
        ),
        "gpt-35-turbo": ModelInfo(
            name="gpt-35-turbo",
            provider="azure_openai",
            max_tokens=4096,
            max_context_length=16385,
            supports_functions=True,
            supports_vision=False,
            supports_streaming=True,
            cost_per_1k_prompt_tokens=0.0005,
            cost_per_1k_completion_tokens=0.0015,
            tags=["fast", "affordable", "legacy"],
        ),
    }

    @property
    def name(self) -> str:
        return "azure_openai"

    def __init__(self, config: Any) -> None:
        """Initialize Azure OpenAI provider with required Azure-specific config."""
        super().__init__(config)

        # Validate required Azure configuration
        if not config.api_key:
            raise ValueError("Azure OpenAI API key is required")

        endpoint = getattr(config, "endpoint", None)
        resource_name = getattr(config, "resource_name", None)

        # TODO: refine this logic. not exactly good read
        if endpoint:
            # Extract from endpoint URL like "https://my-resource.openai.azure.com/"
            self.azure_base_url = endpoint.rstrip("/")
            # Try to extract resource name from endpoint
            if ".openai.azure.com" in endpoint:
                import re

                match = re.match(r"https?://([^.]+)\.openai\.azure\.com", endpoint)
                if match:
                    self.resource_name = match.group(1)
                else:
                    self.resource_name = "azure-openai"
            else:
                self.resource_name = "azure-openai"
        elif resource_name:
            self.resource_name = resource_name
            self.azure_base_url = f"https://{self.resource_name}.openai.azure.com"
        else:
            raise ValueError("Either 'endpoint' or 'resource_name' is required for Azure OpenAI")

        self.api_version = getattr(config, "api_version", "2024-02-15-preview")
        self.deployment_mapping = getattr(config, "deployment_mapping", {})

    def get_available_models(self) -> Dict[str, ModelInfo]:
        return self.MODELS.copy()

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers for Azure OpenAI."""
        headers = {
            "api-key": self.config.api_key or "",
            "Content-Type": "application/json",
        }

        headers.update(self.config.headers)
        return headers

    def _get_deployment_name(self, model: str) -> str:
        """Get Azure deployment name for a model."""
        # Check if user provided custom deployment mapping
        if self.deployment_mapping and model in self.deployment_mapping:
            return str(self.deployment_mapping[model])

        # Default: use model name as deployment name
        # Azure often uses different naming (e.g., gpt-35-turbo instead of gpt-3.5-turbo)
        deployment_name_mapping = {
            "gpt-3.5-turbo": "gpt-35-turbo",
            "gpt-4": "gpt-4",
            "gpt-4-turbo": "gpt-4-turbo",
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
            "gpt-5": "gpt-5",
            "gpt-5-mini": "gpt-5-mini",
            "gpt-5-nano": "gpt-5-nano",
            "gpt-5-chat": "gpt-5-chat",
            "o4-mini": "o4-mini",
            "o3": "o3",
        }

        return deployment_name_mapping.get(model, model)

    def _build_url(self, model: str, stream: bool = False) -> str:
        """Build Azure OpenAI API URL."""
        deployment_name = self._get_deployment_name(model)
        endpoint = "chat/completions"

        url = f"{self.azure_base_url}/openai/deployments/{deployment_name}/{endpoint}"
        url += f"?api-version={self.api_version}"

        return url

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for Azure OpenAI API (same as OpenAI)."""
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

    def _parse_response(self, response_data: Dict[str, Any]) -> AzureOpenAIResponse:
        """Parse Azure OpenAI API response (same format as OpenAI)."""
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

        return AzureOpenAIResponse(
            id=response_data.get("id", ""),
            model=response_data.get("model", ""),
            choices=choices,
            usage=usage,
            created=response_data.get("created"),
            system_fingerprint=response_data.get("system_fingerprint"),
            **raw_response,
        )

    def _parse_streaming_chunk(self, chunk_data: Dict[str, Any]) -> AzureOpenAIResponse:
        """Parse Azure OpenAI streaming chunk (uses delta instead of message)."""
        choices = []

        for choice_data in chunk_data.get("choices", []):
            # Streaming chunks have 'delta' instead of 'message'
            delta_data = choice_data.get("delta", {})

            # Convert delta to message format for consistent interface
            message = Message(
                role=delta_data.get("role", "assistant"),
                content=delta_data.get("content", ""),
                name=delta_data.get("name"),
                function_call=delta_data.get("function_call"),
                tool_calls=delta_data.get("tool_calls"),
            )

            choice = Choice(
                index=choice_data.get("index", 0),
                message=message,
                finish_reason=choice_data.get("finish_reason"),
                logprobs=choice_data.get("logprobs"),
            )
            choices.append(choice)

        # Usage is typically not included in streaming chunks
        usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        # Extract only the keys we want to avoid conflicts
        raw_response = {
            k: v
            for k, v in chunk_data.items()
            if k
            not in [
                "choices",
                "usage",
                "created",
                "id",
                "model",
                "system_fingerprint",
                "provider",
                "cached",
                "raw_response",
            ]
        }

        return AzureOpenAIResponse(
            id=chunk_data.get("id", ""),
            choices=choices,
            created=chunk_data.get("created", 0),
            model=chunk_data.get("model", ""),
            usage=usage,
            system_fingerprint=chunk_data.get("system_fingerprint"),
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
        url = self._build_url(model)

        # Remove model from payload since it's in the URL path
        payload = {
            "messages": self._format_messages(messages),
            **kwargs,
        }

        timeout = getattr(self.config, "timeout", 30) or 30
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                url,
                json=payload,
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                raise ProviderError(
                    f"Azure OpenAI API error: {response.status_code} - {response.text}"
                )

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
        url = self._build_url(model)

        # Remove model from payload since it's in the URL path
        payload = {
            "messages": self._format_messages(messages),
            **kwargs,
        }

        timeout = getattr(self.config, "timeout", 30) or 30
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json=payload,
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                raise ProviderError(
                    f"Azure OpenAI API error: {response.status_code} - {response.text}"
                )

            return self._parse_response(response.json())

    def stream(
        self,
        messages: List[Message],
        model: str,
        **kwargs: Any,
    ) -> Iterator[BaseResponse]:
        """Synchronous streaming completion."""
        url = self._build_url(model, stream=True)

        payload = {
            "messages": self._format_messages(messages),
            "stream": True,
            **kwargs,
        }

        timeout = getattr(self.config, "timeout", 30) or 30
        with httpx.Client(timeout=timeout) as client, client.stream(
            "POST",
            url,
            json=payload,
            headers=self._get_headers(),
        ) as response:
            if response.status_code != 200:
                raise ProviderError(f"Azure OpenAI API error: {response.status_code}")

            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        import json

                        chunk = json.loads(data)
                        yield self._parse_streaming_chunk(chunk)
                    except json.JSONDecodeError:
                        continue

    async def astream(
        self,
        messages: List[Message],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[BaseResponse]:
        """Asynchronous streaming completion."""
        url = self._build_url(model, stream=True)

        payload = {
            "messages": self._format_messages(messages),
            "stream": True,
            **kwargs,
        }

        timeout = getattr(self.config, "timeout", 30) or 30
        async with httpx.AsyncClient(timeout=timeout) as client, client.stream(
            "POST",
            url,
            json=payload,
            headers=self._get_headers(),
        ) as response:
            if response.status_code != 200:
                raise ProviderError(f"Azure OpenAI API error: {response.status_code}")

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        import json

                        chunk = json.loads(data)
                        yield self._parse_streaming_chunk(chunk)
                    except json.JSONDecodeError:
                        continue
