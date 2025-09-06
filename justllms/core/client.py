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
    ):
        self.config = self._load_config(config)
        self.providers = providers if providers is not None else {}
        self.router = router or Router(self.config.routing)
        self.default_model = default_model
        self.default_provider = default_provider

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
                        # Silently skip failed provider initialization
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
        """Create a completion with intelligent routing."""
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
            cached=False,
            **response.raw_response,
        )
