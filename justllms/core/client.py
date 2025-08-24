"""Main client class for JustLLMs."""

from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from justllms.analytics import AnalyticsDashboard
from justllms.cache import CacheManager
from justllms.config import Config
from justllms.core.base import BaseProvider
from justllms.core.completion import Completion, CompletionResponse
from justllms.core.models import Message, ProviderConfig
from justllms.exceptions import ProviderError
from justllms.health import EndpointHealthChecker
from justllms.monitoring import Monitor
from justllms.routing import Router
from justllms.validation import BusinessRuleEngine, ValidationConfig

if TYPE_CHECKING:
    pass


class Client:
    """Main client for interacting with LLM providers."""

    def __init__(
        self,
        config: Optional[Union[str, Dict[str, Any], Config]] = None,
        providers: Optional[Dict[str, BaseProvider]] = None,
        router: Optional[Router] = None,
        cache_manager: Optional[CacheManager] = None,
        monitor: Optional[Monitor] = None,
        conversation_manager: Optional[Any] = None,
        default_model: Optional[str] = None,
        default_provider: Optional[str] = None,
    ):
        self.config = self._load_config(config)
        self.providers = providers if providers is not None else {}
        self.router = router or Router(self.config.routing)
        self.cache_manager = cache_manager or CacheManager(self.config.cache)
        self.monitor = monitor or Monitor(self.config.monitoring)
        self.default_model = default_model
        self.default_provider = default_provider

        # Initialize conversation management
        if conversation_manager is not None:
            self.conversations = conversation_manager
        else:
            # Lazy import to avoid circular dependency
            from justllms.conversations import ConversationManager

            self.conversations = ConversationManager(
                storage_config=getattr(self.config, "conversations", {}), client=self
            )

        self.retrieval = None

        # Initialize analytics dashboard
        self.analytics = AnalyticsDashboard(
            metrics_collector=(
                self.monitor.metrics_collector
                if hasattr(self.monitor, "metrics_collector")
                else None
            ),
            cost_tracker=getattr(self.monitor, "cost_tracker", None),
        )

        # Initialize health checker
        self.health = EndpointHealthChecker(client=self)

        # Initialize validation system
        validation_config = getattr(self.config, "validation", ValidationConfig())
        self.validation = BusinessRuleEngine(validation_config)

        self.completion = Completion(self)

        if providers is None:
            self._initialize_providers()

    def _load_config(self, config: Optional[Union[str, Dict[str, Any], Config]]) -> Config:
        """Load configuration."""
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
        """Initialize providers based on configuration."""
        from justllms.providers import get_provider_class

        for provider_name, provider_config in self.config.providers.items():
            if provider_config.get("enabled", True) and provider_config.get("api_key"):
                provider_class = get_provider_class(provider_name)
                if provider_class:
                    try:
                        config = ProviderConfig(name=provider_name, **provider_config)
                        self.providers[provider_name] = provider_class(config)
                    except Exception:
                        # Log but don't fail if a provider can't be initialized
                        # Warning: Could not initialize provider
                        pass

    def add_provider(self, name: str, provider: BaseProvider) -> None:
        """Add a provider to the client."""
        self.providers[name] = provider

    def get_provider(self, name: str) -> Optional[BaseProvider]:
        """Get a provider by name."""
        return self.providers.get(name)

    def list_providers(self) -> List[str]:
        """List available providers."""
        return list(self.providers.keys())

    def list_models(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """List available models."""
        models = {}

        if provider:
            if provider in self.providers:
                models[provider] = self.providers[provider].get_available_models()
        else:
            for name, prov in self.providers.items():
                models[name] = prov.get_available_models()

        return models

    def _create_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Create a completion."""
        # Route first to get provider and model for monitoring
        if not provider:
            provider, model = self.router.route(
                messages=messages,
                model=model,
                providers=self.providers,
                **kwargs,
            )

        # Ensure model is not None
        if not model:
            raise ValueError("Model is required")

        # Start monitoring with proper provider/model info
        request_id = self.monitor.start_request(
            provider=provider,
            model=model,
            request_type="completion",
            metadata={"messages": messages},
        )

        try:
            # Check if cache should be bypassed
            bypass_cache = kwargs.pop("_bypass_cache", False)

            if not bypass_cache:
                cache_key = self.cache_manager.get_cache_key(messages, model, provider, **kwargs)
                cached_response = self.cache_manager.get(cache_key)

                if cached_response:
                    self.monitor.record_cache_hit(request_id, cache_key)
                    return cached_response
                else:
                    # Record cache miss
                    self.monitor.record_cache_miss(request_id, cache_key)
            else:
                cache_key = None

            if provider not in self.providers:
                raise ProviderError(f"Provider '{provider}' not found")

            prov = self.providers[provider]
            response = prov.complete(messages=messages, model=model, **kwargs)

            # Calculate estimated cost if usage is available
            if response.usage:
                estimated_cost = prov.estimate_cost(response.usage, model)  # type: ignore
                if estimated_cost is not None:
                    response.usage.estimated_cost = estimated_cost

            completion_response = CompletionResponse(
                id=response.id,
                model=response.model,
                choices=response.choices,
                usage=response.usage,
                created=response.created,
                system_fingerprint=response.system_fingerprint,
                provider=provider,
                cached=False,
                **response.raw_response,
            )

            # Only cache if not bypassed and cache_key exists
            if cache_key is not None:
                self.cache_manager.set(cache_key, completion_response)
            self.monitor.end_request(request_id, completion_response)

            return completion_response

        except Exception as e:
            self.monitor.record_error(request_id, e)
            raise

    async def _acreate_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Create an async completion."""
        # Route first to get provider and model for monitoring
        if not provider:
            provider, model = await self.router.aroute(
                messages=messages,
                model=model,
                providers=self.providers,
                **kwargs,
            )

        # Ensure model is not None
        if not model:
            raise ValueError("Model is required")

        # Start monitoring with proper provider/model info
        request_id = self.monitor.start_request(
            provider=provider,
            model=model,
            request_type="completion",
            metadata={"messages": messages},
        )

        try:
            # Check if cache should be bypassed
            bypass_cache = kwargs.pop("_bypass_cache", False)

            if not bypass_cache:
                cache_key = self.cache_manager.get_cache_key(messages, model, provider, **kwargs)
                cached_response = await self.cache_manager.aget(cache_key)

                if cached_response:
                    self.monitor.record_cache_hit(request_id, cache_key)
                    return cached_response
                else:
                    # Record cache miss
                    self.monitor.record_cache_miss(request_id, cache_key)
            else:
                cache_key = None

            if provider not in self.providers:
                raise ProviderError(f"Provider '{provider}' not found")

            prov = self.providers[provider]
            response = await prov.acomplete(messages=messages, model=model, **kwargs)

            # Calculate estimated cost if usage is available
            if response.usage:
                estimated_cost = prov.estimate_cost(response.usage, model)
                if estimated_cost is not None:
                    response.usage.estimated_cost = estimated_cost

            completion_response = CompletionResponse(
                id=response.id,
                model=response.model,
                choices=response.choices,
                usage=response.usage,
                created=response.created,
                system_fingerprint=response.system_fingerprint,
                provider=provider,
                cached=False,
                **response.raw_response,
            )

            # Only cache if not bypassed and cache_key exists
            if cache_key is not None:
                await self.cache_manager.aset(cache_key, completion_response)
            self.monitor.end_request(request_id, completion_response)

            return completion_response

        except Exception as e:
            self.monitor.record_error(request_id, e)
            raise

    def _stream_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[CompletionResponse]:
        """Stream a completion."""
        request_id = self.monitor.start_request()

        try:
            if not provider:
                provider, model = self.router.route(
                    messages=messages,
                    model=model,
                    providers=self.providers,
                    **kwargs,
                )

            # Ensure model is not None
            if not model:
                raise ValueError("Model is required")

            if provider not in self.providers:
                raise ProviderError(f"Provider '{provider}' not found")

            prov = self.providers[provider]

            for response in prov.stream(messages=messages, model=model, **kwargs):
                yield CompletionResponse(
                    id=response.id,
                    model=response.model,
                    choices=response.choices,
                    usage=response.usage,
                    created=response.created,
                    system_fingerprint=response.system_fingerprint,
                    provider=provider,
                    cached=False,
                    **response.raw_response,
                )

            self.monitor.end_streaming_request(request_id)

        except Exception as e:
            self.monitor.record_error(request_id, e)
            raise

    async def _astream_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[CompletionResponse]:
        """Stream an async completion."""
        request_id = self.monitor.start_request()

        try:
            if not provider:
                provider, model = await self.router.aroute(
                    messages=messages,
                    model=model,
                    providers=self.providers,
                    **kwargs,
                )

            # Ensure model is not None
            if not model:
                raise ValueError("Model is required")

            if provider not in self.providers:
                raise ProviderError(f"Provider '{provider}' not found")

            prov = self.providers[provider]

            async for response in prov.astream(messages=messages, model=model, **kwargs):
                yield CompletionResponse(
                    id=response.id,
                    model=response.model,
                    choices=response.choices,
                    usage=response.usage,
                    created=response.created,
                    system_fingerprint=response.system_fingerprint,
                    provider=provider,
                    cached=False,
                    **response.raw_response,
                )

            self.monitor.end_streaming_request(request_id)

        except Exception as e:
            self.monitor.record_error(request_id, e)
            raise
