from justllms.__version__ import __version__
from justllms.core.client import Client
from justllms.core.completion import Completion, CompletionResponse
from justllms.core.models import Message, Role
from justllms.exceptions import JustLLMsError, ProviderError
from justllms.tools import GoogleCodeExecution, GoogleSearch, Tool, ToolRegistry, tool

JustLLM = Client

__all__ = [
    "__version__",
    "JustLLM",
    "Client",
    "Completion",
    "CompletionResponse",
    "Message",
    "Role",
    "JustLLMsError",
    "ProviderError",
    "tool",
    "Tool",
    "ToolRegistry",
    "GoogleSearch",
    "GoogleCodeExecution",
]
