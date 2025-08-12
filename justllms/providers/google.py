"""Google Gemini provider implementation."""

import time
from typing import Any, AsyncIterator, Dict, Iterator, List

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from justllms.core.base import BaseProvider, BaseResponse
from justllms.core.models import Choice, Message, ModelInfo, Role, Usage
from justllms.exceptions import ProviderError


class GoogleResponse(BaseResponse):
    """Google-specific response implementation."""

    pass


class GoogleProvider(BaseProvider):
    """Google Gemini provider implementation."""

    MODELS = {
        "gemini-2.5-pro": ModelInfo(
            name="gemini-2.5-pro",
            provider="google",
            max_tokens=65536,
            max_context_length=1048576,
            supports_functions=True,
            supports_vision=True,
            supports_streaming=True,
            cost_per_1k_prompt_tokens=0.00125,
            cost_per_1k_completion_tokens=0.005,
            tags=[
                "flagship",
                "multimodal",
                "long-context",
                "complex-reasoning",
                "code-analysis",
                "pdf",
            ],
        ),
        "gemini-2.5-flash": ModelInfo(
            name="gemini-2.5-flash",
            provider="google",
            max_tokens=65536,
            max_context_length=1048576,
            supports_functions=True,
            supports_vision=True,
            supports_streaming=True,
            cost_per_1k_prompt_tokens=0.000075,
            cost_per_1k_completion_tokens=0.0003,
            tags=["latest", "multimodal", "long-context", "adaptive-thinking", "cost-efficient"],
        ),
        "gemini-2.5-flash-lite": ModelInfo(
            name="gemini-2.5-flash-lite",
            provider="google",
            max_tokens=65536,
            max_context_length=1048576,
            supports_functions=True,
            supports_vision=True,
            supports_streaming=True,
            cost_per_1k_prompt_tokens=0.00005,
            cost_per_1k_completion_tokens=0.0002,
            tags=["cost-efficient", "high-throughput", "multimodal", "long-context"],
        ),
        "gemini-1.5-pro": ModelInfo(
            name="gemini-1.5-pro",
            provider="google",
            max_tokens=8192,
            max_context_length=2097152,
            supports_functions=True,
            supports_vision=True,
            supports_streaming=True,
            cost_per_1k_prompt_tokens=0.00125,
            cost_per_1k_completion_tokens=0.005,
            tags=["reasoning", "multimodal", "long-context"],
        ),
        "gemini-1.5-flash": ModelInfo(
            name="gemini-1.5-flash",
            provider="google",
            max_tokens=8192,
            max_context_length=1048576,
            supports_functions=True,
            supports_vision=True,
            supports_streaming=True,
            cost_per_1k_prompt_tokens=0.000075,
            cost_per_1k_completion_tokens=0.0003,
            tags=["fast", "efficient", "multimodal", "long-context"],
        ),
        "gemini-1.5-flash-8b": ModelInfo(
            name="gemini-1.5-flash-8b",
            provider="google",
            max_tokens=8192,
            max_context_length=1048576,
            supports_functions=True,
            supports_vision=True,
            supports_streaming=True,
            cost_per_1k_prompt_tokens=0.0000375,
            cost_per_1k_completion_tokens=0.00015,
            tags=["fastest", "affordable", "multimodal"],
        ),
    }

    @property
    def name(self) -> str:
        return "google"

    def get_available_models(self) -> Dict[str, ModelInfo]:
        return self.MODELS.copy()

    def _get_api_endpoint(self, model: str, stream: bool = False) -> str:
        """Get the API endpoint for a model."""
        base_url = self.config.api_base or "https://generativelanguage.googleapis.com"
        method = "streamGenerateContent" if stream else "generateContent"
        return f"{base_url}/v1beta/models/{model}:{method}"

    def _format_messages(self, messages: List[Message]) -> Dict[str, Any]:
        """Format messages for Gemini API."""
        # Gemini uses a different format than OpenAI
        # System instructions are separate
        system_instruction = None
        contents = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                # Gemini uses "user" and "model" (not "assistant")
                role = "user" if msg.role == "user" else "model"

                # Handle content format
                if isinstance(msg.content, str):
                    parts = [{"text": msg.content}]
                else:
                    # Handle multimodal content
                    parts = []
                    for item in msg.content:
                        if item.get("type") == "text":
                            parts.append({"text": item.get("text", "")})
                        elif item.get("type") == "image":
                            # Handle image data
                            image_data = item.get("image", {})
                            if isinstance(image_data, dict):
                                parts.append(
                                    {
                                        "inline_data": {
                                            "mime_type": str(
                                                image_data.get("mime_type", "image/jpeg")
                                            ),
                                            "data": str(image_data.get("data", "")),
                                        }  # type: ignore
                                    }
                                )

                contents.append({"role": role, "parts": parts})

        # Build request
        request_data: Dict[str, Any] = {"contents": contents}

        if system_instruction:
            request_data["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        return request_data

    def _format_generation_config(self, **kwargs: Any) -> Dict[str, Any]:
        """Format generation configuration for Gemini."""
        config = {}

        # Map common parameters to Gemini format
        if "temperature" in kwargs:
            config["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            config["maxOutputTokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            config["topP"] = kwargs["top_p"]
        if "top_k" in kwargs:
            config["topK"] = kwargs["top_k"]
        if "stop" in kwargs:
            config["stopSequences"] = (
                kwargs["stop"] if isinstance(kwargs["stop"], list) else [kwargs["stop"]]
            )

        return config

    def _parse_response(self, response_data: Dict[str, Any], model: str) -> GoogleResponse:
        """Parse Gemini API response."""
        candidates = response_data.get("candidates", [])

        if not candidates:
            raise ProviderError("No candidates in Gemini response")

        # Get the first candidate
        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        # Extract text from parts
        text_content = ""
        for part in parts:
            if "text" in part:
                text_content += part["text"]

        # Create message
        message = Message(
            role=Role.ASSISTANT,
            content=text_content,
        )

        # Create choice
        choice = Choice(
            index=0,
            message=message,
            finish_reason=candidate.get("finishReason", "stop").lower(),
        )

        # Parse usage metadata
        usage_metadata = response_data.get("usageMetadata", {})
        usage = Usage(
            prompt_tokens=usage_metadata.get("promptTokenCount", 0),
            completion_tokens=usage_metadata.get("candidatesTokenCount", 0),
            total_tokens=usage_metadata.get("totalTokenCount", 0),
        )

        # Extract only the keys we want to avoid conflicts
        raw_response = {
            k: v
            for k, v in response_data.items()
            if k not in ["id", "model", "choices", "usage", "created"]
        }

        return GoogleResponse(
            id=response_data.get("id", f"gemini-{int(time.time())}"),
            model=model,
            choices=[choice],
            usage=usage,
            created=int(time.time()),
            **raw_response,
        )

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "Content-Type": "application/json",
        }

    def _get_params(self) -> Dict[str, str]:
        """Get query parameters."""
        return {
            "key": self.config.api_key or "",
        }

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
        url = self._get_api_endpoint(model, stream=False)

        # Format request
        request_data = self._format_messages(messages)

        # Add generation config
        generation_config = self._format_generation_config(**kwargs)
        if generation_config:
            request_data["generationConfig"] = generation_config

        # Add safety settings if provided
        if "safety_settings" in kwargs:
            request_data["safetySettings"] = kwargs["safety_settings"]

        with httpx.Client(timeout=self.config.timeout) as client:
            response = client.post(
                url,
                json=request_data,
                headers=self._get_headers(),
                params=self._get_params(),
            )

            if response.status_code != 200:
                raise ProviderError(f"Gemini API error: {response.status_code} - {response.text}")

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
        url = self._get_api_endpoint(model, stream=False)

        # Format request
        request_data = self._format_messages(messages)

        # Add generation config
        generation_config = self._format_generation_config(**kwargs)
        if generation_config:
            request_data["generationConfig"] = generation_config

        # Add safety settings if provided
        if "safety_settings" in kwargs:
            request_data["safetySettings"] = kwargs["safety_settings"]

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                url,
                json=request_data,
                headers=self._get_headers(),
                params=self._get_params(),
            )

            if response.status_code != 200:
                raise ProviderError(f"Gemini API error: {response.status_code} - {response.text}")

            return self._parse_response(response.json(), model)

    def stream(  # noqa: C901
        self,
        messages: List[Message],
        model: str,
        **kwargs: Any,
    ) -> Iterator[BaseResponse]:
        """Synchronous streaming completion."""
        url = self._get_api_endpoint(model, stream=True)

        # Format request
        request_data = self._format_messages(messages)

        # Add generation config
        generation_config = self._format_generation_config(**kwargs)
        if generation_config:
            request_data["generationConfig"] = generation_config

        # Add safety settings if provided
        if "safety_settings" in kwargs:
            request_data["safetySettings"] = kwargs["safety_settings"]

        # For Gemini streaming, we need to send alt=sse parameter
        params = self._get_params()
        params["alt"] = "sse"

        with httpx.Client(timeout=self.config.timeout) as client, client.stream(
            "POST",
            url,
            json=request_data,
            headers=self._get_headers(),
            params=params,
        ) as response:
            if response.status_code != 200:
                raise ProviderError(
                    f"Gemini API error: {response.status_code} - {response.read().decode('utf-8', errors='ignore')}"
                )

            # Buffer for accumulating partial lines
            buffer = ""

            for chunk in response.iter_bytes():
                buffer += chunk.decode("utf-8", errors="ignore")

                # Process complete lines
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()

                    # Skip empty lines
                    if not line:
                        continue

                    # Skip SSE data prefix
                    if line.startswith("data: "):
                        line = line[6:]

                    # Skip non-JSON lines
                    if not line or line == "[DONE]":
                        continue

                    try:
                        import json

                        chunk_data = json.loads(line)

                        # Extract text from the chunk
                        candidates = chunk_data.get("candidates", [])
                        if candidates:
                            candidate = candidates[0]
                            content = candidate.get("content", {})
                            parts = content.get("parts", [])

                            for part in parts:
                                if "text" in part:
                                    message = Message(role=Role.ASSISTANT, content=part["text"])
                                    choice = Choice(
                                        index=0,
                                        message=message,
                                        finish_reason=candidate.get("finishReason"),
                                    )
                                    yield GoogleResponse(
                                        id=f"gemini-stream-{int(time.time())}",
                                        model=model,
                                        choices=[choice],
                                        created=int(time.time()),
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
        url = self._get_api_endpoint(model, stream=True)

        # Format request
        request_data = self._format_messages(messages)

        # Add generation config
        generation_config = self._format_generation_config(**kwargs)
        if generation_config:
            request_data["generationConfig"] = generation_config

        # Add safety settings if provided
        if "safety_settings" in kwargs:
            request_data["safetySettings"] = kwargs["safety_settings"]

        # For Gemini streaming, we need to send alt=sse parameter
        params = self._get_params()
        params["alt"] = "sse"

        async with httpx.AsyncClient(timeout=self.config.timeout) as client, client.stream(
            "POST",
            url,
            json=request_data,
            headers=self._get_headers(),
            params=params,
        ) as response:
            if response.status_code != 200:
                content = await response.aread()
                raise ProviderError(
                    f"Gemini API error: {response.status_code} - {content.decode()}"
                )

            # Buffer for accumulating partial lines
            buffer = ""

            async for chunk in response.aiter_bytes():
                buffer += chunk.decode("utf-8", errors="ignore")

                # Process complete lines
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()

                    # Skip empty lines
                    if not line:
                        continue

                    # Skip SSE data prefix
                    if line.startswith("data: "):
                        line = line[6:]

                    # Skip non-JSON lines
                    if not line or line == "[DONE]":
                        continue

                    try:
                        import json

                        chunk_data = json.loads(line)

                        # Extract text from the chunk
                        candidates = chunk_data.get("candidates", [])
                        if candidates:
                            candidate = candidates[0]
                            content = candidate.get("content", {})
                            parts = content.get("parts", [])

                            for part in parts:
                                if "text" in part:
                                    message = Message(role=Role.ASSISTANT, content=part["text"])
                                    choice = Choice(
                                        index=0,
                                        message=message,
                                        finish_reason=candidate.get("finishReason"),
                                    )
                                    yield GoogleResponse(
                                        id=f"gemini-stream-{int(time.time())}",
                                        model=model,
                                        choices=[choice],
                                        created=int(time.time()),
                                    )
                    except json.JSONDecodeError:
                        continue
