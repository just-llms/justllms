import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from justllms.cache import (
    BaseCacheBackend,
    DiskCache,
    InMemoryCache,
    ResponseCache,
    _response_from_cache_dict,
    _response_to_cache_dict,
)
from justllms.config import Config
from justllms.core.base import BaseProvider, BaseResponse
from justllms.core.completion import Completion, CompletionResponse
from justllms.core.models import Message, ProviderConfig
from justllms.exceptions import ConfigurationError, ProviderError
from justllms.monitoring import BudgetManager, UsageTracker
from justllms.reliability import FallbackExecutor
from justllms.routing import Router

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from justllms.core.streaming import SyncStreamResponse


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

        # Usage tracking is always on (negligible overhead); budget
        # enforcement only when configured.
        self.usage = UsageTracker()
        self.budget: Optional[BudgetManager] = (
            BudgetManager(self.config.budget, self.usage) if self.config.budget else None
        )

        from justllms.tools.registry import ToolRegistry

        self.tool_registry = ToolRegistry()

        self.cache: Optional[ResponseCache] = self._initialize_cache()

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

    def _initialize_cache(self) -> Optional[ResponseCache]:
        """Build the response cache from configuration.

        Returns:
            ResponseCache instance when caching is enabled in config,
            None otherwise.

        Raises:
            ConfigurationError: If an unknown cache backend is configured.
        """
        cache_config = getattr(self.config, "cache", None)
        if cache_config is None or not cache_config.enabled:
            return None

        backend: BaseCacheBackend
        if cache_config.backend == "memory":
            backend = InMemoryCache(max_size=cache_config.max_size)
        elif cache_config.backend == "disk":
            backend = DiskCache(path=cache_config.path)
        else:
            raise ConfigurationError(
                f"Unknown cache backend '{cache_config.backend}'. "
                "Valid backends are 'memory' and 'disk'.",
                config_key="cache.backend",
                config_value=cache_config.backend,
            )

        return ResponseCache(backend=backend, default_ttl=cache_config.ttl)

    def _initialize_providers(self) -> None:
        """Initialize providers based on configuration settings.

        Creates provider instances for all enabled providers in the configuration
        that have valid API keys (or do not require one). Logs warnings for
        misconfigured providers and raises if every eligible provider fails.

        Raises:
            ConfigurationError: If all eligible providers fail to initialize.
            ImportError: If required provider class cannot be imported.
        """
        from justllms.providers import get_provider_class

        init_failures: List[Tuple[str, Exception]] = []
        attempted_providers: List[str] = []

        for provider_name, provider_config in self.config.providers.items():
            if not provider_config.get("enabled", True):
                continue

            provider_class = get_provider_class(provider_name)
            if not provider_class:
                logger.warning("Unknown provider '%s' in config, skipping", provider_name)
                continue

            requires_key = getattr(provider_class, "requires_api_key", True)
            if requires_key and not provider_config.get("api_key"):
                continue

            attempted_providers.append(provider_name)

            try:
                config = ProviderConfig(name=provider_name, **provider_config)
                self.providers[provider_name] = provider_class(config)
            except Exception as exc:
                logger.warning(
                    "Failed to initialize provider '%s': %s",
                    provider_name,
                    exc,
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
                init_failures.append((provider_name, exc))

        if attempted_providers and not self.providers:
            failure_details = "; ".join(f"{name}: {error}" for name, error in init_failures)
            raise ConfigurationError(
                "Failed to initialize any configured providers. "
                f"{failure_details or 'No failure details available.'}"
            )

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

    def _complete_with_fallback(
        self,
        messages: List[Message],
        provider_name: str,
        selected_model: str,
        fallback_override: Optional[bool],
        **kwargs: Any,
    ) -> CompletionResponse:
        """Run a non-streaming completion with optional failure-driven failover.

        Args:
            messages: List of conversation messages to process.
            provider_name: Primary provider to attempt first.
            selected_model: Primary model to attempt first.
            fallback_override: Per-request failover override. False disables
                failover, True forces it, None uses the routing configuration.
            **kwargs: Additional parameters passed to the provider's complete method.

        Returns:
            CompletionResponse from the first provider in the chain that succeeds,
            with fallback metadata populated when failover engaged.
        """
        executor = FallbackExecutor(self.config.routing)
        use_fallback = fallback_override if fallback_override is not None else executor.enabled

        def _call(name: str, model_name: str) -> CompletionResponse:
            provider_instance = self.providers[name]
            response = provider_instance.complete(messages=messages, model=model_name, **kwargs)
            self._estimate_and_set_cost(response, provider_instance, model_name)
            self.usage.record(name, model_name, response.usage)
            return self._wrap_completion_response(response, name)

        if not use_fallback:
            return _call(provider_name, selected_model)

        chain = executor.build_chain((provider_name, selected_model), self.providers)
        completion, attempts = executor.execute(chain, _call)

        if any(not attempt.succeeded for attempt in attempts):
            completion.fallback_used = True
            completion.fallback_attempts = [attempt.to_dict() for attempt in attempts]

        return completion

    def _make_stream_usage_callback(
        self, provider_name: str, model: str
    ) -> "Callable[[CompletionResponse], None]":
        """Build an on_complete callback that records streaming usage.

        Args:
            provider_name: Provider used for the streaming request.
            model: Model used for the streaming request.

        Returns:
            Callback for SyncStreamResponse.on_complete that records the
            final response's usage in the client's UsageTracker.
        """

        def _on_complete(final_response: "CompletionResponse") -> None:
            self.usage.record(provider_name, model, final_response.usage, streamed=True)

        return _on_complete

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get aggregated usage and cost totals for this client.

        Returns:
            Dict with total_cost, total_tokens, request_count and
            per-provider/per-model breakdowns.
        """
        return self.usage.get_summary()

    def reset_usage(self) -> None:
        """Reset accumulated usage (also resets budget accounting)."""
        self.usage.reset()

    def _create_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> "CompletionResponse | SyncStreamResponse":
        """Create a completion with automatic fallback support.

        Uses configured fallback provider/model or first available provider
        if no specific model is requested. When a fallback chain is configured
        (routing.fallback_chain or routing.auto_fallback), non-streaming
        non-tool requests automatically fail over to the next provider/model
        on retryable errors.

        Args:
            messages: List of conversation messages to process.
            model: Optional specific model to use. Can be model name or
                  'provider/model' format.
            provider: Optional specific provider to use. Overrides fallback selection.
            stream: If True, returns streaming response instead of CompletionResponse.
            **kwargs: Additional parameters passed to the provider's complete method.
                     Common parameters: temperature, max_tokens, top_p, etc.
                     'fallback' (Optional[bool]) is consumed here as a per-request
                     failover override and never forwarded to providers.

        Returns:
            CompletionResponse or StreamResponse depending on stream parameter.

        Raises:
            ValueError: If model is not specified and no fallback is configured,
                       or if tools are requested together with stream=True.
            ProviderError: If the specified provider is not available or if the
                          completion request fails, or if streaming is requested
                          but the provider doesn't support it.
        """
        # Pre-flight budget check: blocks this request if a limit was already
        # reached by previous requests (the crossing request itself completes).
        if self.budget is not None:
            self.budget.check()

        # Per-request failover override; popped so it never reaches provider payloads.
        fallback_override: Optional[bool] = kwargs.pop("fallback", None)

        # Pop cache-control kwargs first so they never reach provider payloads.
        cache_flag = kwargs.pop("cache", None)
        cache_ttl = kwargs.pop("cache_ttl", None)
        if cache_flag is True and self.cache is None:
            raise ConfigurationError(
                "cache=True was passed but response caching is not configured. "
                "Enable it via config, e.g. {'cache': {'enabled': True}}.",
                config_key="cache.enabled",
            )

        # Check if tools are provided
        tools = kwargs.pop("tools", None)
        if tools and stream:
            raise ValueError(
                "Tool calling is not supported with stream=True. "
                "Use stream=False to enable tool execution, or omit tools for streaming."
            )
        if tools:
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

            # Call tool-enabled completion. Usage tracking counts the final
            # response only; intermediate tool-loop LLM calls inside
            # Completion._create_with_tools are not recorded individually.
            tool_response = self.completion._create_with_tools(
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
            self.usage.record(provider_name, selected_model, tool_response.usage)
            return tool_response

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
                stream_response = provider_instance.stream(
                    messages=messages, model=selected_model, **kwargs
                )
                stream_response.on_complete = self._make_stream_usage_callback(
                    provider, selected_model
                )
                return stream_response
            else:
                # Non-streaming with specified provider; explicit provider is the
                # primary and the configured fallback chain still applies after it.
                return self._complete_non_streaming(
                    provider_name=provider,
                    model=selected_model,
                    messages=messages,
                    use_cache=cache_flag is not False,
                    cache_ttl=cache_ttl,
                    fallback_override=fallback_override,
                    **kwargs,
                )

        if stream:
            provider_name, selected_model = self.router.route_streaming(
                messages=messages, providers=self.providers, model=model, **kwargs
            )

            # Stream with routed provider
            provider_instance = self.providers[provider_name]
            stream_response = provider_instance.stream(
                messages=messages, model=selected_model, **kwargs
            )
            stream_response.on_complete = self._make_stream_usage_callback(
                provider_name, selected_model
            )
            return stream_response
        else:
            # Non-streaming route
            provider_name, selected_model = self.router.route(
                messages=messages, model=model, providers=self.providers, **kwargs
            )

            return self._complete_non_streaming(
                provider_name=provider_name,
                model=selected_model,
                messages=messages,
                use_cache=cache_flag is not False,
                cache_ttl=cache_ttl,
                fallback_override=fallback_override,
                **kwargs,
            )

    def _complete_non_streaming(
        self,
        provider_name: str,
        model: str,
        messages: List[Message],
        use_cache: bool = True,
        cache_ttl: Optional[float] = None,
        fallback_override: Optional[bool] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Execute a non-streaming completion with caching and failover.

        The cache is consulted before any provider call (a hit skips failover
        entirely and consumes no budget); on a miss the request runs through
        the failure-driven fallback chain and the successful response is
        cached under the originally requested provider/model key.

        Args:
            provider_name: Resolved provider name.
            model: Resolved model identifier.
            messages: Conversation messages.
            use_cache: Whether the cache may be consulted/updated for this
                call. Effective only when a cache is configured.
            cache_ttl: Optional per-request TTL override in seconds.
            fallback_override: Per-request failover override passed through
                to the fallback executor.
            **kwargs: Provider generation parameters.

        Returns:
            CompletionResponse from the cache (cached=True) or a provider.
        """
        cache_enabled = use_cache and self.cache is not None

        if cache_enabled and self.cache is not None:
            cached_data = self.cache.get(provider_name, model, messages, kwargs)
            if cached_data is not None:
                logger.debug(
                    "Response cache hit for provider '%s', model '%s'", provider_name, model
                )
                return _response_from_cache_dict(cached_data)

        completion_response = self._complete_with_fallback(
            messages=messages,
            provider_name=provider_name,
            selected_model=model,
            fallback_override=fallback_override,
            **kwargs,
        )

        if cache_enabled and self.cache is not None:
            self.cache.set(
                provider_name,
                model,
                messages,
                kwargs,
                _response_to_cache_dict(completion_response),
                ttl=cache_ttl,
            )

        return completion_response
