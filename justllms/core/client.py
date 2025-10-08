from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from justllms.config import Config
from justllms.core.base import BaseProvider, BaseResponse
from justllms.core.completion import Completion, CompletionResponse
from justllms.core.models import Message, ProviderConfig
from justllms.exceptions import ProviderError
from justllms.routing import Router

if TYPE_CHECKING:
    from justllms.core.streaming import AsyncStreamResponse, SyncStreamResponse


class Client:
    """Multi-provider LLM client with automatic fallbacks."""

    def __init__(
        self,
        config: Optional[Union[str, Dict[str, Any], Config]] = None,
        providers: Optional[Dict[str, BaseProvider]] = None,
        router: Optional[Router] = None,
        default_model: Optional[str] = None,
        default_provider: Optional[str] = None,
    ):
        self.config = self._load_config(config)

        self.providers = providers if providers is not None else {}
        self.router = router or Router(self.config.routing)
        self.default_model = default_model
        self.default_provider = default_provider

        from justllms.tools.registry import ToolRegistry

        self.tool_registry = ToolRegistry()

        self.completion = Completion(self)

        if providers is None:
            self._initialize_providers()

    def _load_config(self, config: Optional[Union[str, Dict[str, Any], Config]]) -> Config:
        """Load and validate configuration from various sources.

        Args:
            config: Configuration object, dictionary, file path, or None for defaults.
                   Can be a Config instance, dict with config values, string path to
                   config file, or None to load from environment/defaults.

        Returns:
            Config: Validated configuration object with provider settings and fallbacks.

        Raises:
            FileNotFoundError: If config file path is provided but file doesn't exist.
            ValueError: If config format is invalid.
        """
        if isinstance(config, Config):
            return config
        elif isinstance(config, dict):
            return Config(**config)
        elif isinstance(config, str):
            return Config.from_file(config)
        else:
            # Load default config with environment variables
            from justllms.config import load_config

            return load_config(use_defaults=True, use_env=True)

    def _initialize_providers(self) -> None:
        """Initialize providers based on configuration settings.

        Creates provider instances for all enabled providers in the configuration
        that have valid API keys. Silently skips providers that fail to initialize
        to allow partial functionality when some providers are misconfigured.

        Raises:
            ImportError: If required provider class cannot be imported.
        """
        from justllms.providers import get_provider_class

        for provider_name, provider_config in self.config.providers.items():
            if not provider_config.get("enabled", True):
                continue

            provider_class = get_provider_class(provider_name)
            if not provider_class:
                continue

            requires_key = getattr(provider_class, "requires_api_key", True)
            if requires_key and not provider_config.get("api_key"):
                continue

            try:
                config = ProviderConfig(name=provider_name, **provider_config)
                self.providers[provider_name] = provider_class(config)
            except Exception:
                pass

    def add_provider(self, name: str, provider: BaseProvider) -> None:
        """Add a provider instance to the client.

        Args:
            name: Unique identifier for the provider (e.g., 'openai', 'anthropic').
            provider: Configured provider instance implementing BaseProvider.
        """
        self.providers[name] = provider

    def get_provider(self, name: str) -> Optional[BaseProvider]:
        """Retrieve a provider instance by name.

        Args:
            name: Provider identifier to look up.

        Returns:
            Optional[BaseProvider]: Provider instance if found, None otherwise.
        """
        return self.providers.get(name)

    def list_providers(self) -> List[str]:
        """Get names of all available providers.

        Returns:
            List[str]: List of provider names that are currently initialized
                      and available for use.
        """
        return list(self.providers.keys())

    def register_tools(self, tools: List[Any]) -> None:
        """Register tools for use with completions.

        Args:
            tools: List of Tool instances to register.
        """
        from justllms.tools.models import Tool

        for tool in tools:
            if isinstance(tool, Tool):
                self.tool_registry.register(tool)
            elif hasattr(tool, "tool"):
                # It's a decorated function
                self.tool_registry.register(tool.tool)
            else:
                raise ValueError(f"Invalid tool type: {type(tool)}")

    def list_models(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get available models from providers.

        Args:
            provider: Optional provider name to filter models. If None, returns
                     models from all available providers.

        Returns:
            Dict[str, Any]: Dictionary mapping provider names to their available
                           models. Each model entry contains ModelInfo details.
        """
        models = {}

        if provider:
            if provider in self.providers:
                models[provider] = self.providers[provider].get_available_models()
        else:
            for name, prov in self.providers.items():
                models[name] = prov.get_available_models()

        return models

    def _estimate_and_set_cost(
        self, response: BaseResponse, provider_instance: BaseProvider, model: str
    ) -> None:
        """Estimate cost and set it on response.usage if available.

        Args:
            response: Provider response with usage data.
            provider_instance: Provider instance for cost estimation.
            model: Model identifier used for the request.
        """
        if response.usage:
            estimated_cost = provider_instance.estimate_cost(response.usage, model)
            if estimated_cost is not None:
                response.usage.estimated_cost = estimated_cost

    def _wrap_completion_response(
        self, response: BaseResponse, provider: str
    ) -> CompletionResponse:
        """Wrap provider response in CompletionResponse with provider name.

        Args:
            response: Provider response to wrap.
            provider: Provider name to include in response.

        Returns:
            CompletionResponse with provider metadata.
        """
        return CompletionResponse(
            id=response.id,
            model=response.model,
            choices=response.choices,
            usage=response.usage,
            created=response.created,
            system_fingerprint=response.system_fingerprint,
            provider=provider,
            **response.raw_response,
        )

    def _create_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> "CompletionResponse | SyncStreamResponse | AsyncStreamResponse":
        """Create a completion with automatic fallback support.

        Uses configured fallback provider/model or first available provider
        if no specific model is requested.

        Args:
            messages: List of conversation messages to process.
            model: Optional specific model to use. Can be model name or
                  'provider/model' format.
            provider: Optional specific provider to use. Overrides fallback selection.
            stream: If True, returns streaming response instead of CompletionResponse.
            **kwargs: Additional parameters passed to the provider's complete method.
                     Common parameters: temperature, max_tokens, top_p, etc.

        Returns:
            CompletionResponse or StreamResponse depending on stream parameter.

        Raises:
            ValueError: If model is not specified and no fallback is configured.
            ProviderError: If the specified provider is not available or if the
                          completion request fails, or if streaming is requested
                          but the provider doesn't support it.
        """
        # Check if tools are provided
        tools = kwargs.pop("tools", None)
        if tools and not stream:
            # Route to tool-enabled completion
            # Determine provider/model for tools
            if not provider:
                provider_name, selected_model = self.router.route_with_tools(
                    messages=messages, providers=self.providers, model=model, **kwargs
                )
            else:
                provider_name = provider
                if provider not in self.providers:
                    raise ProviderError(f"Provider '{provider}' not found")

                provider_instance = self.providers[provider]
                models = provider_instance.get_available_models()
                _model: Optional[str] = (
                    model if model else (list(models.keys())[0] if models else None)
                )

                if not _model:
                    raise ValueError(f"No models available for provider {provider}")

                selected_model = _model

            # Extract tool execution params
            tool_choice = kwargs.pop("tool_choice", "auto")
            execute_tools = kwargs.pop(
                "execute_tools", self.config.routing.execute_tools_by_default
            )
            max_iterations = kwargs.pop("max_iterations", self.config.routing.max_tool_iterations)
            timeout = kwargs.pop("timeout", None)

            # Call tool-enabled completion
            return self.completion._create_with_tools(
                messages=messages,
                tools=tools,
                provider=provider_name,
                model=selected_model,
                tool_choice=tool_choice,
                execute_tools=execute_tools,
                max_iterations=max_iterations,
                timeout=timeout,
                **kwargs,
            )

        if provider:
            if provider not in self.providers:
                raise ProviderError(f"Provider '{provider}' not found")

            provider_instance = self.providers[provider]

            # Determine model to use
            if model:
                selected_model = model
            else:
                # Try to get default model from provider
                models = provider_instance.get_available_models()
                if models:
                    selected_model = list(models.keys())[0]
                else:
                    raise ValueError(f"No models available for provider {provider}")

            if stream:
                if not provider_instance.supports_streaming_for_model(selected_model):
                    streaming_providers = [
                        name for name, prov in self.providers.items() if prov.supports_streaming()
                    ]
                    streaming_hint = (
                        f" Try using one of these providers: {', '.join(streaming_providers)}"
                        if streaming_providers
                        else ""
                    )
                    raise ProviderError(
                        f"Provider '{provider}' does not support streaming for model '{selected_model}'. "
                        f"Use stream=False or switch to a streaming-capable provider.{streaming_hint}"
                    )

                # Stream with specified provider
                return provider_instance.stream(messages=messages, model=selected_model, **kwargs)
            else:
                # Non-streaming with specified provider
                response = provider_instance.complete(
                    messages=messages, model=selected_model, **kwargs
                )
                self._estimate_and_set_cost(response, provider_instance, selected_model)
                return self._wrap_completion_response(response, provider)

        if stream:
            provider_name, selected_model = self.router.route_streaming(
                messages=messages, providers=self.providers, model=model, **kwargs
            )

            # Stream with routed provider
            provider_instance = self.providers[provider_name]
            return provider_instance.stream(messages=messages, model=selected_model, **kwargs)
        else:
            # Non-streaming route
            provider_name, selected_model = self.router.route(
                messages=messages, model=model, providers=self.providers, **kwargs
            )

            provider_instance = self.providers[provider_name]
            response = provider_instance.complete(messages=messages, model=selected_model, **kwargs)
            self._estimate_and_set_cost(response, provider_instance, selected_model)
            return self._wrap_completion_response(response, provider_name)
