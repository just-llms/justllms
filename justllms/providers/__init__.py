"""Provider implementations for various LLM services."""

from typing import Dict, List, Optional, Type

from justllms.core.base import BaseProvider

_PROVIDERS: Dict[str, Type[BaseProvider]] = {}


def register_provider(name: str, provider_class: Type[BaseProvider]) -> None:
    """Register a provider class."""
    _PROVIDERS[name.lower()] = provider_class


def get_provider_class(name: str) -> Optional[Type[BaseProvider]]:
    """Get a provider class by name."""
    return _PROVIDERS.get(name.lower())


def list_available_providers() -> List[str]:
    """List all available provider names."""
    return list(_PROVIDERS.keys())


try:
    from justllms.providers.openai import OpenAIProvider

    register_provider("openai", OpenAIProvider)
except ImportError:
    pass

try:
    from justllms.providers.azure_openai import AzureOpenAIProvider

    register_provider("azure_openai", AzureOpenAIProvider)
except ImportError:
    pass

try:
    from justllms.providers.anthropic import AnthropicProvider

    register_provider("anthropic", AnthropicProvider)
    register_provider("claude", AnthropicProvider)
except ImportError:
    pass

try:
    from justllms.providers.google import GoogleProvider

    register_provider("google", GoogleProvider)
    register_provider("gemini", GoogleProvider)
except ImportError:
    pass

try:
    from justllms.providers.xai import XAIProvider

    register_provider("xai", XAIProvider)
    register_provider("grok", XAIProvider)
except ImportError:
    pass

try:
    from justllms.providers.cohere import CohereProvider

    register_provider("cohere", CohereProvider)
except ImportError:
    pass

try:
    from justllms.providers.replicate import ReplicateProvider

    register_provider("replicate", ReplicateProvider)
except ImportError:
    pass

try:
    from justllms.providers.grok import GrokProvider

    register_provider("grok", GrokProvider)
except ImportError:
    pass

try:
    from justllms.providers.deepseek import DeepSeekProvider

    register_provider("deepseek", DeepSeekProvider)
except ImportError:
    pass


__all__ = [
    "register_provider",
    "get_provider_class",
    "list_available_providers",
]
