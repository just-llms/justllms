import json
import logging
import time
from typing import Any, Dict, Iterator, List, Optional

import httpx

from justllms.core.base import BaseProvider, BaseResponse
from justllms.core.models import Choice, Message, ModelInfo, Role, Usage
from justllms.core.streaming import StreamChunk, SyncStreamResponse
from justllms.tools.adapters.base import BaseToolAdapter

logger = logging.getLogger(__name__)


class GoogleResponse(BaseResponse):
    """Google-specific response implementation."""

    pass


class GoogleProvider(BaseProvider):
    """Google Gemini provider implementation."""

    supports_tools = True
    """Gemini supports function calling."""

    supports_native_tools = True
    """Gemini supports native tools like Google Search and Code Execution."""

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

    def _get_api_endpoint(self, model: str, streaming: bool = False) -> str:
        """Get the API endpoint for a model.

        Args:
            model: Model identifier.
            streaming: If True, return streaming endpoint.
        """
        base_url = self.config.api_base or "https://generativelanguage.googleapis.com"
        endpoint_type = "streamGenerateContent" if streaming else "generateContent"
        return f"{base_url}/v1beta/models/{model}:{endpoint_type}"

    def _format_messages(self, messages: List[Message]) -> Dict[str, Any]:
        """Format messages for Gemini API."""
        # Gemini uses a different format than OpenAI
        # System instructions are separate
        system_instruction = None
        contents = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = msg.content
            else:
                # Gemini uses "user" and "model" (not "assistant")
                role = "user" if msg.role == Role.USER else "model"

                # Handle content format
                if isinstance(msg.content, str):
                    parts = [{"text": msg.content}]
                else:
                    # Handle multimodal content and tool calling parts
                    parts = []
                    for item in msg.content:
                        # Check if it's already a properly formatted part (functionCall, functionResponse)
                        if "functionCall" in item or "functionResponse" in item:
                            # Pass through tool calling parts as-is
                            parts.append(item)
                        elif item.get("type") == "text":
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
        """Format generation configuration for Gemini.

        Maps common parameters to Gemini's generationConfig format and merges
        provider-specific generation_config dict.

        Precedence: top-level params > generation_config dict
        """
        config = {}

        param_mapping = {
            "temperature": "temperature",
            "top_p": "topP",
            "top_k": "topK",
            "max_tokens": "maxOutputTokens",
            "presence_penalty": "presencePenalty",
            "frequency_penalty": "frequencyPenalty",
        }

        for snake_case, camel_case in param_mapping.items():
            if snake_case in kwargs and kwargs[snake_case] is not None:
                config[camel_case] = kwargs[snake_case]

        if "stop" in kwargs and kwargs["stop"] is not None:
            config["stopSequences"] = (
                kwargs["stop"] if isinstance(kwargs["stop"], list) else [kwargs["stop"]]
            )

        if "generation_config" in kwargs and kwargs["generation_config"]:
            gemini_config = kwargs["generation_config"]

            for key in gemini_config:
                if key in config:
                    logger.debug(
                        f"Parameter '{key}' specified in both top-level and generation_config. "
                        f"Using top-level value: {config[key]}"
                    )

            for key, value in gemini_config.items():
                if key not in config:
                    config[key] = value

        if "n" in kwargs and kwargs["n"] is not None:
            logger.debug(
                f"Parameter 'n' is OpenAI-specific. For Gemini, use "
                f"generation_config={{'candidateCount': {kwargs['n']}}}. Ignoring 'n'."
            )

        return config

    def _parse_response(self, response_data: Dict[str, Any], model: str) -> GoogleResponse:
        """Parse Gemini API response.

        Handles multiple candidates when candidateCount > 1 is specified.
        Each candidate becomes a separate choice in the response.
        """
        candidates = response_data.get("candidates", [])

        if not candidates:
            from justllms.exceptions import ProviderError

            raise ProviderError("No candidates in Gemini response")

        choices = []
        for idx, candidate in enumerate(candidates):
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

            # Create choice with correct index
            choice = Choice(
                index=idx,
                message=message,
                finish_reason=candidate.get("finishReason", "stop").lower(),
            )
            choices.append(choice)

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
            choices,  # Pass all choices
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
            timeout=timeout,
        )

        return self._parse_response(response_data, model)

    def complete_with_tools(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: str = "",
        tool_choice: Optional[Any] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> BaseResponse:
        """Complete with tool support, including native Google tools.

        Args:
            messages: List of messages for the completion.
            tools: Formatted tool definitions (already formatted by adapter).
            model: Model identifier to use.
            tool_choice: Tool selection configuration (already formatted).
            timeout: Optional timeout in seconds.
            **kwargs: Additional provider-specific parameters.

        Returns:
            BaseResponse with tool call information.
        """
        url = self._get_api_endpoint(model)

        # Format request
        request_data = self._format_messages(messages)

        # Add generation config
        generation_config = self._format_generation_config(**kwargs)
        if generation_config:
            request_data["generationConfig"] = generation_config

        # Add tools - already in Gemini format from adapter
        if tools is not None and tools:
            request_data["tools"] = tools

        # Add tool choice configuration
        # Only add toolConfig if we have ONLY user-defined functions (no native tools)
        # Native tools (google_search, code_execution) don't support toolConfig
        # Mixed (native + user) also doesn't support toolConfig per Gemini live-tools docs
        if tool_choice and tools:
            # Check if there are any native tools
            has_native_tools = any(
                "google_search" in tool_entry or "code_execution" in tool_entry
                for tool_entry in tools
            )

            # Check if there are user functions
            has_function_declarations = any(
                "functionDeclarations" in tool_entry or "function_declarations" in tool_entry
                for tool_entry in tools
            )

            # Only send toolConfig for user-only functions (no native tools)
            if has_function_declarations and not has_native_tools:
                request_data["toolConfig"] = tool_choice

        # Add safety settings if provided
        if "safety_settings" in kwargs:
            request_data["safetySettings"] = kwargs["safety_settings"]

        response_data = self._make_http_request(
            url=url,
            payload=request_data,
            headers=self._get_headers(),
            params=self._get_params(),
            timeout=timeout,
        )

        return self._parse_response(response_data, model)

    def _parse_sse_chunk(self, line: str) -> Optional[StreamChunk]:
        """Parse a single SSE line from Gemini streaming response.

        Args:
            line: Raw SSE line to parse.

        Returns:
            StreamChunk if line contains valid data, None otherwise.
        """
        line = line.strip()
        if not line:
            return None

        # Gemini SSE format: "data: {...}"
        if not line.startswith("data: "):
            return None

        data = line[6:]  # Remove "data: " prefix

        # Ignore [DONE] marker (if sent by API)
        if data == "[DONE]":
            return None

        try:
            chunk_data = json.loads(data)
            candidates = chunk_data.get("candidates", [])

            if not candidates:
                return None

            candidate = candidates[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            # Extract text from parts
            text_content = ""
            for part in parts:
                if "text" in part:
                    text_content += part["text"]

            finish_reason = candidate.get("finishReason")

            # Get usage metadata if available (typically only in final chunk)
            usage_metadata = chunk_data.get("usageMetadata")
            usage = None
            if usage_metadata:
                usage = Usage(
                    prompt_tokens=usage_metadata.get("promptTokenCount", 0),
                    completion_tokens=usage_metadata.get("candidatesTokenCount", 0),
                    total_tokens=usage_metadata.get("totalTokenCount", 0),
                )

            if text_content or finish_reason:
                return StreamChunk(
                    content=text_content if text_content else None,
                    finish_reason=finish_reason.lower() if finish_reason else None,
                    usage=usage,
                    raw=chunk_data,
                )
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse Gemini SSE chunk: {data}")

        return None

    def _stream_gemini_response(
        self,
        url: str,
        payload: Dict[str, Any],
        params: Dict[str, str],
        timeout: Optional[float] = None,
    ) -> Iterator[StreamChunk]:
        """Stream response from Gemini API.

        Args:
            url: API endpoint URL.
            payload: Request payload.
            params: Query parameters (includes API key).
            timeout: Optional timeout in seconds.

        Yields:
            StreamChunk objects from the response.

        Raises:
            ProviderError: If the streaming request fails.
        """
        from justllms.exceptions import ProviderError

        # Add SSE parameter for streaming
        stream_params = {**params, "alt": "sse"}

        try:
            with httpx.Client(timeout=timeout) as client, client.stream(
                "POST",
                url,
                json=payload,
                headers=self._get_headers(),
                params=stream_params,
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    chunk = self._parse_sse_chunk(line)
                    if chunk is not None:
                        yield chunk
        except (httpx.HTTPError, httpx.RequestError) as e:
            raise ProviderError(f"Google streaming request failed: {str(e)}") from e

    def stream(
        self,
        messages: List[Message],
        model: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> SyncStreamResponse:
        """Stream completion using Server-Sent Events.

        Args:
            messages: List of messages for the completion.
            model: Model identifier to use.
            timeout: Optional timeout in seconds.
            **kwargs: Additional provider-specific parameters.

        Returns:
            SyncStreamResponse: Streaming response iterator.
        """
        url = self._get_api_endpoint(model, streaming=True)

        # Format request
        request_data = self._format_messages(messages)

        # Add generation config
        generation_config = self._format_generation_config(**kwargs)
        if generation_config:
            request_data["generationConfig"] = generation_config

        # Add safety settings if provided
        if "safety_settings" in kwargs:
            request_data["safetySettings"] = kwargs["safety_settings"]

        # Use streaming helper
        stream_iter = self._stream_gemini_response(
            url=url,
            payload=request_data,
            params=self._get_params(),
            timeout=timeout,
        )

        return SyncStreamResponse(
            provider=self, model=model, messages=messages, raw_stream=stream_iter
        )

    def supports_streaming(self) -> bool:
        """Google Gemini supports streaming."""
        return True

    def supports_streaming_for_model(self, model: str) -> bool:
        """Check if model supports streaming.

        All Gemini models support streaming.
        """
        return model in self.get_available_models()

    def get_tool_adapter(self) -> Optional[BaseToolAdapter]:
        """Return the Google tool adapter."""
        from justllms.tools.adapters.google import GoogleToolAdapter

        return GoogleToolAdapter()
