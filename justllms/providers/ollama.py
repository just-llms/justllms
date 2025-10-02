from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from justllms.core.base import BaseProvider, BaseResponse
from justllms.core.models import Message, ModelInfo
from justllms.exceptions import ProviderError


class OllamaResponse(BaseResponse):
    """Ollama-specific response implementation."""

    pass


class OllamaProvider(BaseProvider):
    """Provider implementation for locally hosted Ollama models."""

    requires_api_key = False
    _DEFAULT_BASE_URL = "http://localhost:11434"

    _FALLBACK_MODELS: dict[str, ModelInfo] = {
        "llama3.1:70b": ModelInfo(
            name="llama3.1:70b",
            provider="ollama",
            supports_functions=False,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.0,
            cost_per_1k_completion_tokens=0.0,
            tags=["local", "ollama", "flagship"],
        ),
        "llama3.1:8b": ModelInfo(
            name="llama3.1:8b",
            provider="ollama",
            supports_functions=False,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.0,
            cost_per_1k_completion_tokens=0.0,
            tags=["local", "ollama", "efficient"],
        ),
        "mistral": ModelInfo(
            name="mistral",
            provider="ollama",
            supports_functions=False,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.0,
            cost_per_1k_completion_tokens=0.0,
            tags=["local", "ollama", "lightweight"],
        ),
        "phi3:14b": ModelInfo(
            name="phi3:14b",
            provider="ollama",
            supports_functions=False,
            supports_vision=False,
            cost_per_1k_prompt_tokens=0.0,
            cost_per_1k_completion_tokens=0.0,
            tags=["local", "ollama", "reasoning"],
        ),
    }

    def __init__(self, config: Any) -> None:
        """Initialize the Ollama provider.

        Args:
            config: Provider configuration containing optional base_url/api_base,
                   allowed_models, model_overrides, and other Ollama-specific settings.
        """
        super().__init__(config)
        base_url = self.config.api_base or self.config.base_url or self._DEFAULT_BASE_URL
        self._base_url = str(base_url).rstrip("/")

    @property
    def name(self) -> str:
        """Return the provider name.

        Returns:
            The string identifier "ollama" for this provider.
        """
        return "ollama"

    def get_available_models(self) -> dict[str, ModelInfo]:
        """Retrieve all available Ollama models from the local instance.

        Queries the Ollama API to discover installed models. Falls back to a
        hardcoded list if the API is unavailable. Respects allowed_models and
        model_overrides configuration.

        Returns:
            Dictionary mapping model names to ModelInfo objects with metadata,
            pricing (always $0.00 for local models), and capabilities.
        """
        if self._models_cache is not None:
            return self._models_cache.copy()

        models = self._fetch_installed_models()
        if not models:
            models = self._build_fallback_models()

        self._models_cache = models
        return models.copy()

    def complete(
        self, messages: list[Message], model: str, timeout: Any = None, **kwargs: Any
    ) -> BaseResponse:
        """Execute a chat completion request using the Ollama API.

        Args:
            messages: List of conversation messages to send to the model.
            model: Name of the Ollama model to use (e.g., 'llama3.1:8b').
            timeout: Optional timeout in seconds. If None, no timeout is enforced.
            **kwargs: Additional parameters including:
                - temperature: Sampling temperature (0.0-2.0)
                - top_p: Nucleus sampling parameter
                - top_k: Top-k sampling parameter
                - max_tokens: Maximum tokens to generate (maps to num_predict)
                - stop: Stop sequences
                - keep_alive: Model keep-alive duration
                - metadata: Additional metadata for the request
                - options: Additional Ollama-specific options dict
                - presence_penalty, frequency_penalty, repeat_penalty, seed

        Returns:
            BaseResponse containing the model output, usage statistics, and metadata.

        Raises:
            ProviderError: If the Ollama API request fails or returns an error.
        """
        payload = {
            "model": model,
            "messages": self._format_messages_base(messages),
            "stream": kwargs.pop("stream", False),
        }

        stop_sequences = kwargs.pop("stop", None)
        if stop_sequences is not None:
            payload["stop"] = stop_sequences

        keep_alive = kwargs.pop("keep_alive", None) or getattr(self.config, "keep_alive", None)
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        metadata = kwargs.pop("metadata", None) or getattr(self.config, "metadata", None)
        if metadata is not None:
            payload["metadata"] = metadata

        options: dict[str, Any] = {}
        option_keys = {
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "presence_penalty": "presence_penalty",
            "frequency_penalty": "frequency_penalty",
            "repeat_penalty": "repeat_penalty",
            "seed": "seed",
        }
        for kwarg, option_name in option_keys.items():
            value = kwargs.pop(kwarg, None)
            if value is not None:
                options[option_name] = value

        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        extra_options = kwargs.pop("options", None)
        if isinstance(extra_options, dict):
            options.update(extra_options)

        if options:
            payload["options"] = options

        response_data = self._make_http_request(
            url=self._chat_endpoint,
            payload=payload,
            headers=self._get_request_headers(),
            timeout=timeout,
        )

        return self._parse_response(response_data, model)

    @property
    def _chat_endpoint(self) -> str:
        """Return the Ollama chat completion API endpoint URL.

        Returns:
            Full URL to the /api/chat endpoint.
        """
        return f"{self._base_url}/api/chat"

    @property
    def _tags_endpoint(self) -> str:
        """Return the Ollama tags API endpoint URL for model discovery.

        Returns:
            Full URL to the /api/tags endpoint.
        """
        return f"{self._base_url}/api/tags"

    def _fetch_installed_models(self) -> dict[str, ModelInfo]:
        """Fetch currently installed models from the Ollama instance.

        Queries the /api/tags endpoint to discover locally pulled models.
        Filters by allowed_models if configured and applies model_overrides.

        Returns:
            Dictionary of model names to ModelInfo objects, or empty dict if
            Ollama is unavailable or the API call fails.
        """
        try:
            data = self._make_http_request(
                url=self._tags_endpoint,
                payload={},
                headers=self._get_request_headers(),
                method="GET",
            )
        except ProviderError:
            return {}
        except Exception:  # pragma: no cover - defensive
            return {}

        models = {}
        allowed_models = self._get_allowed_models()
        overrides = self._get_model_overrides()

        for entry in data.get("models", []):
            name = entry.get("name") if isinstance(entry, dict) else None
            if not isinstance(name, str):
                continue
            if allowed_models and name not in allowed_models:
                continue
            models[name] = self._construct_model_info(name, overrides.get(name))

        if not models and allowed_models:
            # Handles case where allowed list targets models not yet pulled
            for model_name in allowed_models:
                models[model_name] = self._construct_model_info(
                    model_name, overrides.get(model_name)
                )

        return models

    def _build_fallback_models(self) -> dict[str, ModelInfo]:
        """Build model list from hardcoded fallback models.

        Used when Ollama API is unavailable. Applies allowed_models filter
        and model_overrides configuration.

        Returns:
            Dictionary of model names to ModelInfo objects from _FALLBACK_MODELS.
        """
        allowed_models = self._get_allowed_models()
        overrides = self._get_model_overrides()
        fallback = {}

        for name, info in self._FALLBACK_MODELS.items():
            if allowed_models and name not in allowed_models:
                continue
            fallback[name] = info.model_copy()

        for name, override in overrides.items():
            if allowed_models and name not in allowed_models:
                continue
            fallback[name] = self._construct_model_info(name, override)

        return {name: model for name, model in fallback.items()}

    def _construct_model_info(self, name: str, override: dict[str, Any] | None) -> ModelInfo:
        """Construct a ModelInfo object for a given model name.

        Args:
            name: The model name (e.g., 'llama3.1:8b').
            override: Optional dictionary containing custom model metadata from
                     config.model_overrides to override default values.

        Returns:
            ModelInfo object with model metadata, capabilities, and pricing.
        """
        override = override or {}
        base_tags = ["local", "ollama"]
        tags_value = override.get("tags")
        extra_tags = tags_value if isinstance(tags_value, list) else []
        tags = base_tags + [tag for tag in extra_tags if tag not in base_tags]

        return ModelInfo(
            name=name,
            provider=self.name,
            max_tokens=override.get("max_tokens"),
            max_context_length=override.get("max_context_length"),
            supports_functions=override.get("supports_functions", False),
            supports_vision=override.get("supports_vision", False),
            cost_per_1k_prompt_tokens=override.get("cost_per_1k_prompt_tokens", 0.0),
            cost_per_1k_completion_tokens=override.get("cost_per_1k_completion_tokens", 0.0),
            latency_ms_per_token=override.get("latency_ms_per_token"),
            tags=tags,
        )

    def _get_allowed_models(self) -> set[str] | None:
        """Get the set of allowed model names from configuration.

        Returns:
            Set of allowed model name strings if config.allowed_models is set,
            otherwise None (meaning all models are allowed).
        """
        allowed = getattr(self.config, "allowed_models", None)
        if isinstance(allowed, (list, tuple, set)):
            return {str(model_name) for model_name in allowed}
        return None

    def _get_model_overrides(self) -> dict[str, dict[str, Any]]:
        """Get model-specific configuration overrides.

        Returns:
            Dictionary mapping model names to override dicts containing custom
            metadata (e.g., max_tokens, supports_vision, custom pricing).
        """
        overrides = getattr(self.config, "model_overrides", None)
        if isinstance(overrides, dict):
            return overrides
        return {}

    def _parse_response(self, response_data: dict[str, Any], model: str) -> BaseResponse:
        """Parse the Ollama API response into a standardized BaseResponse.

        Args:
            response_data: Raw response dictionary from Ollama /api/chat endpoint.
            model: The model name used for the request.

        Returns:
            BaseResponse object with standardized format, including choices,
            usage statistics, and metadata.
        """
        message_data = response_data.get("message", {})
        choice = self._create_standard_choice(
            {**message_data, "finish_reason": "stop"},
            index=0,
        )

        prompt_tokens = int(response_data.get("prompt_eval_count") or 0)
        completion_tokens = int(response_data.get("eval_count") or 0)
        usage_payload = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        usage = self._create_standard_usage(usage_payload)

        created_at = self._parse_created_timestamp(response_data.get("created_at"))

        response_payload = {
            **response_data,
            "id": response_data.get("id") or f"ollama-{uuid4()}",
            "model": response_data.get("model") or model,
            "choices": [
                {
                    "index": 0,
                    "message": message_data,
                    "finish_reason": "stop",
                }
            ],
            "usage": usage_payload,
            "created": created_at,
        }

        return self._create_base_response(
            OllamaResponse,
            response_payload,
            [choice],
            usage,
            model,
        )

    def _parse_created_timestamp(self, value: Any) -> int | None:
        """Parse Ollama's ISO timestamp string to Unix epoch seconds.

        Args:
            value: Timestamp value from Ollama response (ISO 8601 string or other).

        Returns:
            Unix timestamp as integer, or None if parsing fails or value is invalid.
        """
        if not isinstance(value, str):
            return None
        try:
            normalized = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
            return int(dt.timestamp())
        except ValueError:
            return None

    def _get_request_headers(self) -> dict[str, str]:
        """Get HTTP headers for Ollama API requests.

        Returns:
            Dictionary of HTTP headers including Content-Type and any custom
            headers from configuration.
        """
        return self._get_default_headers()
