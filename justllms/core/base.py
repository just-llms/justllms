"""Base classes for providers and responses."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from justllms.core.models import Choice, Message, ModelInfo, ProviderConfig, Usage


class BaseResponse:
    """Base class for all provider responses."""

    def __init__(
        self,
        id: str,
        model: str,
        choices: List[Choice],
        usage: Optional[Usage] = None,
        created: Optional[int] = None,
        system_fingerprint: Optional[str] = None,
        **kwargs: Any,
    ):
        self.id = id
        self.model = model
        self.choices = choices
        self.usage = usage
        self.created = created
        self.system_fingerprint = system_fingerprint
        self.raw_response = kwargs

    @property
    def content(self) -> Optional[str]:
        """Get the content of the first choice."""
        if self.choices and self.choices[0].message.content:
            content = self.choices[0].message.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list) and content:
                return str(content[0].get("text", ""))
        return None

    @property
    def message(self) -> Optional[Message]:
        """Get the message of the first choice."""
        if self.choices:
            return self.choices[0].message
        return None


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._models_cache: Optional[Dict[str, ModelInfo]] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        pass

    @abstractmethod
    async def acomplete(
        self,
        messages: List[Message],
        model: str,
        **kwargs: Any,
    ) -> BaseResponse:
        """Async completion method."""
        pass

    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        model: str,
        **kwargs: Any,
    ) -> BaseResponse:
        """Sync completion method."""
        pass

    @abstractmethod
    def astream(
        self,
        messages: List[Message],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[BaseResponse]:
        """Async streaming completion method."""
        pass

    @abstractmethod
    def stream(
        self,
        messages: List[Message],
        model: str,
        **kwargs: Any,
    ) -> Iterator[BaseResponse]:
        """Sync streaming completion method."""
        pass

    @abstractmethod
    def get_available_models(self) -> Dict[str, ModelInfo]:
        """Get available models for this provider."""
        pass

    def validate_model(self, model: str) -> bool:
        """Validate if a model is available."""
        models = self.get_available_models()
        return model in models

    def get_model_info(self, model: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        models = self.get_available_models()
        return models.get(model)

    def estimate_cost(self, usage: Usage, model: str) -> Optional[float]:
        """Estimate the cost for the given usage."""
        model_info = self.get_model_info(model)
        if not model_info or not model_info.cost_per_1k_prompt_tokens:
            return None

        prompt_cost = (usage.prompt_tokens / 1000) * model_info.cost_per_1k_prompt_tokens
        completion_cost = (usage.completion_tokens / 1000) * (
            model_info.cost_per_1k_completion_tokens or 0
        )

        return prompt_cost + completion_cost
