import time
from typing import Any, Dict, List

from justllms.core.base import BaseProvider, BaseResponse
from justllms.core.models import Choice, Message, ModelInfo, Role, Usage


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

    def _get_api_endpoint(self, model: str) -> str:
        """Get the API endpoint for a model."""
        base_url = self.config.api_base or "https://generativelanguage.googleapis.com"
        return f"{base_url}/v1beta/models/{model}:generateContent"

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
            from justllms.exceptions import ProviderError

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

        if "id" not in response_data:
            response_data["id"] = f"gemini-{int(time.time())}"
        if "created" not in response_data:
            response_data["created"] = int(time.time())

        return self._create_base_response(  # type: ignore[return-value]
            GoogleResponse,
            response_data,
            [choice],
            usage,
            model,
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

    def complete(
        self,
        messages: List[Message],
        model: str,
        **kwargs: Any,
    ) -> BaseResponse:
        """Synchronous completion."""
        url = self._get_api_endpoint(model)

        # Format request
        request_data = self._format_messages(messages)

        # Add generation config
        generation_config = self._format_generation_config(**kwargs)
        if generation_config:
            request_data["generationConfig"] = generation_config

        # Add safety settings if provided
        if "safety_settings" in kwargs:
            request_data["safetySettings"] = kwargs["safety_settings"]

        response_data = self._make_http_request(
            url=url,
            payload=request_data,
            headers=self._get_headers(),
            params=self._get_params(),
        )

        return self._parse_response(response_data, model)
