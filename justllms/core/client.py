from typing import Any, Dict, List, Optional, Union

from justllms.config import Config
from justllms.core.base import BaseProvider
from justllms.core.completion import Completion, CompletionResponse
from justllms.core.models import Message, ProviderConfig
from justllms.exceptions import ProviderError
from justllms.routing import Router


class Client:
    """Simplified client focused on intelligent routing."""

    def __init__(
        self,
        config: Optional[Union[str, Dict[str, Any], Config]] = None,
        providers: Optional[Dict[str, BaseProvider]] = None,
        router: Optional[Router] = None,
        default_model: Optional[str] = None,
        default_provider: Optional[str] = None,
        routing_strategy: Optional[str] = None,
    ):
        self.config = self._load_config(config)

        if routing_strategy:
            self.config.routing.strategy = routing_strategy

        self.providers = providers if providers is not None else {}
        self.router = router or Router(self.config.routing)
        self.default_model = default_model
        self.default_provider = default_provider

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
            Config: Validated configuration object with provider and routing settings.

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

    def _create_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Create a completion using intelligent routing to select optimal provider.

        Uses the configured routing strategy to automatically select the best
        provider and model combination based on the request characteristics,
        unless specific provider/model are requested.

        Args:
            messages: List of conversation messages to process.
            model: Optional specific model to use. Can be model name or
                  'provider/model' format.
            provider: Optional specific provider to use. Overrides routing.
            **kwargs: Additional parameters passed to the provider's complete method.
                     Common parameters: temperature, max_tokens, top_p, etc.

        Returns:
            CompletionResponse: Response object containing the generated completion,
                              usage statistics, and provider metadata.

        Raises:
            ValueError: If model is not specified and cannot be determined by routing.
            ProviderError: If the specified provider is not available or if the
                          completion request fails.
        """
        # Use intelligent routing to select provider and model
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
        response = prov.complete(messages=messages, model=model, **kwargs)

        # Calculate estimated cost if usage is available
        if response.usage:
            estimated_cost = prov.estimate_cost(response.usage, model)
            if estimated_cost is not None:
                response.usage.estimated_cost = estimated_cost

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
