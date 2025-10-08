from typing import Any, Dict, List, Optional, Tuple, Union

from justllms.core.base import BaseProvider
from justllms.core.models import Message
from justllms.exceptions import ProviderError


class Router:
    """Simple provider and model selector with fallback support.

    Handles model selection logic:
    1. If model explicitly specified (e.g., "provider/model"), use it
    2. Else if fallback configured, use fallback
    3. Else use first available provider/model
    """

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], Any]] = None,
        fallback_provider: Optional[str] = None,
        fallback_model: Optional[str] = None,
    ):
        """Initialize the router.

        Args:
            config: Optional config dict or RoutingConfig object.
            fallback_provider: Optional fallback provider name.
            fallback_model: Optional fallback model name.
        """
        # Handle both dict and RoutingConfig object
        if config is not None and hasattr(config, "model_dump"):
            # It's a Pydantic model, convert to dict
            self.config = config.model_dump()
        else:
            self.config = config or {}

        # Get fallback values from config if not provided
        self.fallback_provider = fallback_provider or self.config.get("fallback_provider")
        self.fallback_model = fallback_model or self.config.get("fallback_model")

    def route(  # noqa: C901
        self,
        messages: List[Message],
        model: Optional[str] = None,
        providers: Optional[Dict[str, BaseProvider]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        """Select provider and model using fallback logic.

        Args:
            messages: The messages to process (unused in selection).
            model: Optional specific model requested.
            providers: Available providers.
            constraints: Additional constraints (unused in selection).
            **kwargs: Additional parameters.

        Returns:
            Tuple of (provider_name, model_name)

        Raises:
            ValueError: If no providers or suitable models available.
        """
        if not providers:
            raise ValueError("No providers available")

        # If specific model requested, try to find it
        if model:
            # Check if it's in format "provider/model"
            if "/" in model:
                provider_name, model_name = model.split("/", 1)
                if provider_name not in providers:
                    raise ValueError(f"Provider '{provider_name}' not found")

                provider = providers[provider_name]
                if not provider.validate_model(model_name):
                    raise ValueError(
                        f"Model '{model_name}' not found in provider '{provider_name}'"
                    )

                return provider_name, model_name

            # Check all providers for the model
            for provider_name, provider in providers.items():
                if provider.validate_model(model):
                    return provider_name, model

            raise ValueError(f"Model '{model}' not found in any available provider")

        # No specific model requested - use fallback or first available
        # First, try configured fallback if provided
        if self.fallback_provider and self.fallback_model and self.fallback_provider in providers:
            provider = providers[self.fallback_provider]
            available_models = provider.get_available_models()
            if self.fallback_model in available_models:
                return self.fallback_provider, self.fallback_model

        # Fall back to first available provider and model
        for provider_name, provider in providers.items():
            models = provider.get_available_models()
            if models:
                model_name = list(models.keys())[0]
                return provider_name, model_name

        raise ValueError("No models available in any provider")

    def route_streaming(
        self,
        messages: List[Message],
        providers: Dict[str, BaseProvider],
        model: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        """Select provider and model for streaming requests.

        Filters providers to only those supporting streaming before selection.

        Args:
            messages: The messages to process.
            providers: Available providers.
            model: Optional specific model requested.
            constraints: Additional constraints for routing.
            **kwargs: Additional parameters.

        Returns:
            Tuple of (provider_name, model_name).

        Raises:
            ValueError: If no streaming-capable providers available.
        """
        # Filter to streaming-capable providers
        streaming_providers = {
            name: provider for name, provider in providers.items() if provider.supports_streaming()
        }

        if not streaming_providers:
            raise ProviderError(
                "No streaming-capable providers configured. "
                "Enable openai, azure_openai, google, or ollama providers, or set stream=False."
            )

        # If specific model requested, validate streaming support
        if model:
            # Use normal route logic first
            provider_name, model_name = self.route(
                messages,
                model=model,
                providers=streaming_providers,
                constraints=constraints,
                **kwargs,
            )

            # Validate model supports streaming
            provider = streaming_providers[provider_name]
            if not provider.supports_streaming_for_model(model_name):
                raise ProviderError(
                    f"Model '{model_name}' does not support streaming on provider '{provider_name}'. "
                    f"Use stream=False or choose a different model."
                )

            return provider_name, model_name

        # Route among streaming providers
        provider_name, model_name = self.route(
            messages, providers=streaming_providers, constraints=constraints, **kwargs
        )

        # Double-check model supports streaming
        provider = streaming_providers[provider_name]
        if not provider.supports_streaming_for_model(model_name):
            # Try to find another model from this provider that supports streaming
            for available_model in provider.get_available_models():
                if provider.supports_streaming_for_model(available_model):
                    return provider_name, available_model

            raise ProviderError(
                f"Model '{model_name}' does not support streaming on provider '{provider_name}'. "
                f"Use stream=False or choose a different model."
            )

        return provider_name, model_name

    def route_with_tools(
        self,
        messages: List[Message],
        providers: Dict[str, BaseProvider],
        model: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        """Select provider and model for tool calling requests.

        Filters providers to only those supporting tools before selection.

        Args:
            messages: The messages to process.
            providers: Available providers.
            model: Optional specific model requested.
            constraints: Additional constraints for routing.
            **kwargs: Additional parameters.

        Returns:
            Tuple of (provider_name, model_name).

        Raises:
            ValueError: If no tool-capable providers available.
        """
        # Filter to tool-capable providers
        tool_providers = {
            name: provider for name, provider in providers.items() if provider.supports_tools
        }

        if not tool_providers:
            raise ProviderError(
                "No tool-capable providers configured. "
                "Enable openai, anthropic, google, or azure_openai providers."
            )

        # If specific model requested, route to it
        if model:
            provider_name, model_name = self.route(
                messages,
                model=model,
                providers=tool_providers,
                constraints=constraints,
                **kwargs,
            )
            return provider_name, model_name

        # Route among tool-capable providers
        provider_name, model_name = self.route(
            messages, providers=tool_providers, constraints=constraints, **kwargs
        )

        return provider_name, model_name
