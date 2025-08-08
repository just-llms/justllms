"""Conversation models and data structures."""

import time
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from justllms.core.models import Message, Role, Usage


class ConversationState(str, Enum):
    """Conversation state enumeration."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class ConversationMessage(BaseModel):
    """A message in a conversation with metadata."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    timestamp: float = Field(default_factory=time.time)
    model: Optional[str] = Field(default=None, description="Model that generated this message")
    provider: Optional[str] = Field(
        default=None, description="Provider that generated this message"
    )
    usage: Optional[Usage] = Field(default=None, description="Token usage for this message")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_message(self) -> Message:
        """Convert to core Message object."""
        return Message(role=Role(self.role), content=self.content)

    @classmethod
    def from_message(cls, message: Message, **kwargs: Any) -> "ConversationMessage":
        """Create from core Message object."""
        # Convert complex content to string for conversation storage
        content_str = message.content if isinstance(message.content, str) else str(message.content)
        return cls(role=message.role.value, content=content_str, **kwargs)


class ConversationConfig(BaseModel):
    """Configuration for a conversation."""

    # Model settings
    default_model: Optional[str] = Field(
        default=None, description="Default model for the conversation"
    )
    default_provider: Optional[str] = Field(
        default=None, description="Default provider for the conversation"
    )
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt for the conversation"
    )

    # Context management
    max_context_tokens: Optional[int] = Field(
        default=8000, description="Maximum context window tokens"
    )
    context_strategy: str = Field(
        default="truncate", description="Context management strategy: truncate, summarize, compress"
    )
    context_compression_ratio: float = Field(
        default=0.7, description="Target compression ratio when using compress strategy"
    )
    keep_system_prompt: bool = Field(
        default=True, description="Always keep system prompt in context"
    )

    # Conversation behavior
    auto_save: bool = Field(default=True, description="Automatically save conversation state")
    auto_title: bool = Field(default=True, description="Automatically generate conversation title")
    enable_analytics: bool = Field(default=True, description="Track conversation analytics")

    # Storage settings
    storage_backend: str = Field(
        default="memory", description="Storage backend: memory, disk, redis"
    )
    storage_config: Dict[str, Any] = Field(
        default_factory=dict, description="Storage backend configuration"
    )

    # Advanced features
    enable_branching: bool = Field(default=False, description="Allow conversation branching")
    enable_multi_model: bool = Field(
        default=True, description="Allow switching models within conversation"
    )

    class Config:
        extra = "allow"


class ConversationSummary(BaseModel):
    """Summary information about a conversation."""

    id: str
    title: Optional[str] = None
    state: ConversationState = ConversationState.ACTIVE
    created_at: float
    updated_at: float
    message_count: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    models_used: List[str] = Field(default_factory=list)
    providers_used: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationAnalytics(BaseModel):
    """Analytics data for a conversation."""

    conversation_id: str

    # Usage metrics
    total_messages: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    system_messages: int = 0

    # Token metrics
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Cost metrics
    total_cost: float = 0.0
    cost_by_model: Dict[str, float] = Field(default_factory=dict)
    cost_by_provider: Dict[str, float] = Field(default_factory=dict)

    # Performance metrics
    average_response_time: float = 0.0
    response_times: List[float] = Field(default_factory=list)

    # Context metrics
    context_truncations: int = 0
    context_compressions: int = 0
    max_context_used: int = 0

    # Quality metrics (can be extended)
    quality_scores: List[float] = Field(default_factory=list)
    average_quality_score: Optional[float] = None

    # Temporal metrics
    conversation_duration: float = 0.0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def update_from_message(self, message: ConversationMessage, response_time: float = 0.0) -> None:
        """Update analytics from a new message."""
        self.total_messages += 1
        self.updated_at = time.time()

        if message.role == "user":
            self.user_messages += 1
        elif message.role == "assistant":
            self.assistant_messages += 1
        elif message.role == "system":
            self.system_messages += 1

        if message.usage:
            self.total_tokens += message.usage.total_tokens
            self.prompt_tokens += message.usage.prompt_tokens
            self.completion_tokens += message.usage.completion_tokens

            if message.usage.estimated_cost:
                self.total_cost += message.usage.estimated_cost

                if message.model:
                    self.cost_by_model[message.model] = (
                        self.cost_by_model.get(message.model, 0.0) + message.usage.estimated_cost
                    )

                if message.provider:
                    self.cost_by_provider[message.provider] = (
                        self.cost_by_provider.get(message.provider, 0.0)
                        + message.usage.estimated_cost
                    )

        if response_time > 0:
            self.response_times.append(response_time)
            self.average_response_time = sum(self.response_times) / len(self.response_times)

    def calculate_conversation_duration(self, start_time: float) -> None:
        """Calculate total conversation duration."""
        self.conversation_duration = self.updated_at - start_time


class ConversationBranch(BaseModel):
    """A branch in a conversation tree."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    parent_message_id: Optional[str] = None
    name: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    messages: List[ConversationMessage] = Field(default_factory=list)
    is_active: bool = Field(True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
