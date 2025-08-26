"""Main router class for intelligent model selection."""

from typing import Any, Dict, List, Optional, Tuple, Union

from justllms.core.base import BaseProvider
from justllms.core.models import Message
from justllms.routing.strategies import (
    ClusterBasedStrategy,
    CostOptimizedStrategy,
    LatencyOptimizedStrategy,
    QualityOptimizedStrategy,
    RoutingStrategy,
    TaskBasedStrategy,
)


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
            self.strategy = self._create_strategy(strategy or self.config.get("strategy", "cost"))

    def _create_strategy(self, strategy_name: str) -> RoutingStrategy:
        """Create a routing strategy from name."""
        strategy_configs = self.config.get("strategy_configs", {})

        if strategy_name == "cluster":
            config = strategy_configs.get("cluster", {})
            return ClusterBasedStrategy(**config)
        elif strategy_name == "cost":
            config = strategy_configs.get("cost", {})
            return CostOptimizedStrategy(**config)
        elif strategy_name == "latency":
            config = strategy_configs.get("latency", {})
            return LatencyOptimizedStrategy(**config)
        elif strategy_name == "quality":
            config = strategy_configs.get("quality", {})
            return QualityOptimizedStrategy(**config)
        elif strategy_name == "task":
            return TaskBasedStrategy()
        else:
            # Default to cost optimization
            return CostOptimizedStrategy()

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

            # If we get here, the model wasn't found in any provider
            raise ValueError(f"Model '{model}' not found in any available provider")

        # Use routing strategy
        try:
            provider_name, model_name = self.strategy.select(
                messages=messages,
                providers=providers,
                constraints=constraints,
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

    async def aroute(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        providers: Optional[Dict[str, BaseProvider]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        """Async version of route."""
        # For now, routing logic is synchronous
        # In the future, this could do async operations like checking provider health
        return self.route(
            messages=messages,
            model=model,
            providers=providers,
            constraints=constraints,
            **kwargs,
        )

    def set_strategy(self, strategy: Union[str, RoutingStrategy]) -> None:
        """Set the routing strategy."""
        if isinstance(strategy, RoutingStrategy):
            self.strategy = strategy
        else:
            self.strategy = self._create_strategy(strategy)

    def get_strategy_name(self) -> str:
        """Get the name of the current strategy."""
        return self.strategy.__class__.__name__
