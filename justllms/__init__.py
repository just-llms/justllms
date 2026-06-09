from justllms.__version__ import __version__
from justllms.core.client import Client
from justllms.core.completion import Completion, CompletionResponse
from justllms.core.models import Message, Role
from justllms.exceptions import BudgetExceededError, JustLLMsError, ProviderError
from justllms.monitoring import BudgetConfig, BudgetManager, UsageRecord, UsageTracker
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
    "BudgetExceededError",
    "BudgetConfig",
    "BudgetManager",
    "UsageRecord",
    "UsageTracker",
    "tool",
    "Tool",
    "ToolRegistry",
    "GoogleSearch",
    "GoogleCodeExecution",
]
