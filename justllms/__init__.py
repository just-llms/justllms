"""JustLLMs - A unified gateway for multiple LLM providers.

This package provides:
- Unified API interface for multiple LLM providers
- Conversational AI with automatic context management
- Intelligent model routing and selection
- Built-in monitoring, logging, and cost tracking
- Response caching and fallback mechanisms
- Persistent conversation storage (memory, disk, Redis)
- Extensible provider system
"""

from justllms.__version__ import __version__

__author__ = "Your Name"
__email__ = "your.email@example.com"

from justllms.conversations import Conversation, ConversationManager
from justllms.conversations.models import ConversationConfig, ConversationState
from justllms.core.client import Client
from justllms.core.completion import Completion, CompletionResponse
from justllms.core.models import Message, Role
from justllms.exceptions import JustLLMsError, ProviderError, RouteError

# Main client alias
JustLLM = Client

__all__ = [
    "__version__",
    "JustLLM",  # Main client
    "Client",  # Alternative name
    "Completion",
    "CompletionResponse",
    "Message",
    "Role",
    "Conversation",
    "ConversationManager",
    "ConversationConfig",
    "ConversationState",
    "JustLLMsError",
    "ProviderError",
    "RouteError",
]
