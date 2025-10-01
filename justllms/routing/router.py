from typing import Any, Dict, List, Optional, Tuple, Union

from justllms.core.base import BaseProvider
from justllms.core.models import Message
from justllms.routing.strategies import ClusterBasedStrategy, RoutingStrategy


class Router:
    """Router for intelligent model selection."""

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], Any]] = None,
        strategy: Optional[Union[str, RoutingStrategy]] = None,
        fallback_provider: Optional[str] = None,
        fallback_model: Optional[str] = None,
    ):
        # Handle both dict and RoutingConfig object
        if config is not None and hasattr(config, "model_dump"):
            # It's a Pydantic model, convert to dict
            self.config = config.model_dump()
        else:
            self.config = config or {}

        # Get fallback values from config if not provided
        self.fallback_provider = fallback_provider or self.config.get("fallback_provider")
        self.fallback_model = fallback_model or self.config.get("fallback_model")

        # Initialize strategy
        if isinstance(strategy, RoutingStrategy):
            self.strategy = strategy
        else:
            self.strategy = self._create_strategy(
                strategy or self.config.get("strategy", "cluster")
            )

    def _create_strategy(self, strategy_name: str) -> RoutingStrategy:
        """Instantiate a routing strategy based on its name identifier.

        Creates and configures routing strategy instances with settings from
        the router configuration. Only cluster-based routing is supported.

        Args:
            strategy_name: Name of the strategy to create (only 'cluster' is supported).

        Returns:
            RoutingStrategy: Configured strategy instance ready for routing decisions.
        """
        strategy_configs = self.config.get("strategy_configs", {})

        if strategy_name == "cluster":
            config = strategy_configs.get("cluster", {})
            return ClusterBasedStrategy(**config)
        else:
            return ClusterBasedStrategy()

    def route(  # noqa: C901
        self,
        messages: List[Message],
        model: Optional[str] = None,
        providers: Optional[Dict[str, BaseProvider]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        """Route to the best provider and model.

        Args:
            messages: The messages to process
            model: Optional specific model requested
            providers: Available providers
            constraints: Additional constraints for routing
            **kwargs: Additional parameters

        Returns:
            Tuple of (provider_name, model_name)
        """
        if not providers:
            raise ValueError("No providers available for routing")

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

        # Use routing strategy
        try:
            provider_name, model_name = self.strategy.select(
                messages=messages,
                providers=providers,
                constraints=constraints,
                fallback_provider=self.fallback_provider,
                fallback_model=self.fallback_model,
                **kwargs,
            )
            return provider_name, model_name
        except Exception as e:
            # Fallback logic
            if (
                self.fallback_provider
                and self.fallback_model
                and self.fallback_provider in providers
            ):
                provider = providers[self.fallback_provider]
                if provider.validate_model(self.fallback_model):
                    return self.fallback_provider, self.fallback_model

            # Last resort: use first available model
            for provider_name, provider in providers.items():
                models = provider.get_available_models()
                if models:
                    model_name = list(models.keys())[0]
                    return provider_name, model_name

            raise ValueError(f"No suitable model found: {str(e)}") from e

    def set_strategy(self, strategy: Union[str, RoutingStrategy]) -> None:
        """Update the routing strategy used by this router.

        Args:
            strategy: Either a strategy name string or a configured RoutingStrategy
                     instance to use for future routing decisions.
        """
        if isinstance(strategy, RoutingStrategy):
            self.strategy = strategy
        else:
            self.strategy = self._create_strategy(strategy)

    def get_strategy_name(self) -> str:
        """Retrieve the class name of the currently active routing strategy.

        Returns:
            str: Name of the current strategy class (e.g., 'ClusterBasedStrategy').
        """
        return self.strategy.__class__.__name__
