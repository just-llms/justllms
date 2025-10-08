import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from justllms.core.base import DEFAULT_TIMEOUT, BaseProvider, BaseResponse
from justllms.core.models import Choice, Message, ModelInfo, Usage
from justllms.core.streaming import StreamChunk, SyncStreamResponse
from justllms.exceptions import ProviderError
from justllms.tools.adapters.base import BaseToolAdapter

logger = logging.getLogger(__name__)


class AzureOpenAIResponse(BaseResponse):
    """Azure OpenAI-specific response implementation."""

    pass


class AzureOpenAIProvider(BaseProvider):
    """Azure OpenAI provider implementation."""

    supports_tools = True
    """Azure OpenAI supports function calling (same as OpenAI)."""

    # Azure OpenAI models with deployment name mapping
    MODELS = {
        "gpt-5": ModelInfo(
            name="gpt-5",
            provider="azure_openai",
            max_tokens=128000,
            max_context_length=272000,
            supports_functions=True,
            supports_vision=True,
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
            cost_per_1k_prompt_tokens=0.8,
            cost_per_1k_completion_tokens=3.2,
            tags=["chat", "multimodal"],
        ),
        "gpt-4o": ModelInfo(
            name="gpt-4o",
            provider="azure_openai",
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
            provider="azure_openai",
            max_tokens=16384,
            max_context_length=128000,
            supports_functions=True,
            supports_vision=True,
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

        # Support for a single default deployment name
        self.default_deployment = getattr(config, "deployment", None)

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
        """Get Azure deployment name for a model.

        Priority:
        1. Custom deployment_mapping for the specific model
        2. Default deployment (if configured)
        3. Model name fallback with standard conversions
        """
        # Check if user provided custom deployment mapping for this specific model
        if self.deployment_mapping and model in self.deployment_mapping:
            return str(self.deployment_mapping[model])

        # If a default deployment is configured, use it for all models
        if self.default_deployment:
            return str(self.default_deployment)

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

    def _build_url(self, model: str) -> str:
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
        """Synchronous completion with parameter filtering.

        Args:
            messages: List of messages for the completion.
            model: Model identifier to use.
            timeout: Optional timeout in seconds. If None, no timeout is enforced.
            **kwargs: Additional provider-specific parameters.
        """
        url = self._build_url(model)

        supported_params = {
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "n",
            "presence_penalty",
            "frequency_penalty",
            "tools",
            "tool_choice",
            "response_format",
            "seed",
            "user",
            "logprobs",
            "top_logprobs",
            "logit_bias",
        }

        ignored_params = {"top_k", "generation_config", "timeout"}

        payload: Dict[str, Any] = {
            "messages": self._format_messages(messages),
        }

        for key, value in kwargs.items():
            if value is not None:
                if key in ignored_params:
                    logger.debug(f"Parameter '{key}' is not supported by Azure OpenAI. Ignoring.")
                elif key in supported_params:
                    payload[key] = value
                else:
                    logger.debug(f"Unknown parameter '{key}' passed to Azure OpenAI API")
                    payload[key] = value

        from justllms.core.base import DEFAULT_TIMEOUT

        timeout_config = timeout if timeout is not None else DEFAULT_TIMEOUT

        with httpx.Client(timeout=timeout_config) as client:
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

    def _parse_sse_line(self, line: str) -> Optional[StreamChunk]:
        """Parse a single SSE line into a StreamChunk.

        Uses OpenAI-compatible SSE format parsing.

        Args:
            line: Raw SSE line to parse.

        Returns:
            StreamChunk if line contains valid data, None otherwise.
        """
        line = line.strip()
        if not line:
            return None

        if not line.startswith("data: "):
            return None

        data = line[6:]  # Remove "data: " prefix

        if data == "[DONE]":
            return None  # Signal end of stream

        try:
            chunk_data = json.loads(data)
            choices = chunk_data.get("choices", [])

            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                finish_reason = choices[0].get("finish_reason")

                if content or finish_reason:
                    return StreamChunk(
                        content=content,
                        finish_reason=finish_reason,
                        raw=chunk_data,
                    )
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse SSE chunk: {data}")

        return None

    def stream(
        self,
        messages: List[Message],
        model: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> SyncStreamResponse:
        """Stream completion using Server-Sent Events.

        Args:
            messages: Conversation messages to process.
            model: Model identifier for the request.
            timeout: Optional timeout in seconds.
            **kwargs: Additional parameters.

        Returns:
            SyncStreamResponse: Streaming response iterator.
        """
        url = self._build_url(model)

        # Build payload with streaming enabled
        payload: Dict[str, Any] = {
            "messages": self._format_messages(messages),
            "stream": True,
        }

        supported_params = {
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "n",
            "presence_penalty",
            "frequency_penalty",
            "tools",
            "tool_choice",
            "response_format",
            "seed",
            "user",
            "logprobs",
            "top_logprobs",
            "logit_bias",
        }

        ignored_params = {"top_k", "generation_config", "timeout"}

        for key, value in kwargs.items():
            if value is not None:
                if key in ignored_params:
                    logger.debug(f"Parameter '{key}' is not supported. Ignoring.")
                elif key in supported_params:
                    payload[key] = value

        # Use shared SSE parsing helper
        from justllms.core.streaming import parse_sse_stream

        timeout_config = timeout if timeout is not None else DEFAULT_TIMEOUT

        raw_stream = parse_sse_stream(
            url=url,
            payload=payload,
            headers=self._get_headers(),
            parse_chunk_fn=self._parse_sse_line,
            timeout=timeout_config,
            error_prefix="Azure OpenAI streaming request",
        )

        return SyncStreamResponse(
            provider=self, model=model, messages=messages, raw_stream=raw_stream
        )

    def supports_streaming(self) -> bool:
        """Azure OpenAI supports streaming."""
        return True

    def supports_streaming_for_model(self, model: str) -> bool:
        """Check if model supports streaming."""
        return model in self.get_available_models()

    def get_tool_adapter(self) -> Optional[BaseToolAdapter]:
        """Return the Azure tool adapter."""
        from justllms.tools.adapters.azure import AzureToolAdapter

        return AzureToolAdapter()
