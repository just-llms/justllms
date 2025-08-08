"""Common data models used across the package."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class Role(str, Enum):
    """Message role enumeration."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class Message(BaseModel):
    """Unified message format for all providers."""

    model_config = ConfigDict(extra="allow")

    role: Role
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: Optional[float] = None


class Choice(BaseModel):
    """Response choice."""

    index: int
    message: Message
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


class ModelInfo(BaseModel):
    """Information about a model."""

    name: str
    provider: str
    max_tokens: Optional[int] = None
    max_context_length: Optional[int] = None
    supports_functions: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    cost_per_1k_prompt_tokens: Optional[float] = None
    cost_per_1k_completion_tokens: Optional[float] = None
    latency_ms_per_token: Optional[float] = None
    tags: List[str] = Field(default_factory=list)


class ProviderConfig(BaseModel):
    """Configuration for a provider."""

    model_config = ConfigDict(extra="allow")

    name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    organization: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    headers: Dict[str, str] = Field(default_factory=dict)
