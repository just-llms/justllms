"""Core conversation implementation."""

import asyncio
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from justllms.conversations.context import ContextManager
from justllms.conversations.models import (
    ConversationAnalytics,
    ConversationConfig,
    ConversationMessage,
    ConversationState,
    ConversationSummary,
)
from justllms.conversations.storage import ConversationStorage
from justllms.core.models import Message, Role


class Conversation:
    """A conversation session that manages context and state."""

    def __init__(
        self,
        conversation_id: Optional[str] = None,
        config: Optional[ConversationConfig] = None,
        storage: Optional[ConversationStorage] = None,
        client: Optional[Any] = None,  # JustLLM client
    ):
        self.id = conversation_id or str(uuid4())
        self.config = config or ConversationConfig()
        self.storage = storage
        self.client = client

        # Initialize managers
        self.context_manager = ContextManager(self.config)

        # Conversation state
        self.messages: List[ConversationMessage] = []
        self.state = ConversationState.ACTIVE
        self.created_at = time.time()
        self.updated_at = time.time()

        # Analytics
        self.analytics = ConversationAnalytics(conversation_id=self.id)

        # Metadata
        self.title: Optional[str] = None
        self.tags: List[str] = []
        self.metadata: Dict[str, Any] = {}

    @property
    def summary(self) -> ConversationSummary:
        """Get conversation summary."""
        return ConversationSummary(
            id=self.id,
            title=self.title,
            state=self.state,
            created_at=self.created_at,
            updated_at=self.updated_at,
            message_count=len(self.messages),
            total_tokens=self.analytics.total_tokens,
            estimated_cost=self.analytics.total_cost,
            models_used=list(self.analytics.cost_by_model.keys()),
            providers_used=list(self.analytics.cost_by_provider.keys()),
            tags=self.tags,
            metadata=self.metadata,
        )

    def send(
        self,
        content: str,
        role: str = "user",
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> ConversationMessage:
        """Send a message and get a response (synchronous)."""
        return self._run_async(self.send_async(content, role, model, provider, **kwargs))  # type: ignore

    async def send_async(
        self,
        content: str,
        role: str = "user",
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> ConversationMessage:
        """Send a message and get a response (asynchronous)."""
        if self.state != ConversationState.ACTIVE:
            raise ValueError(f"Cannot send message to {self.state.value} conversation")

        # Add user message
        user_message = ConversationMessage(role=role, content=content, timestamp=time.time())

        self.messages.append(user_message)
        self.updated_at = time.time()

        # Update analytics for user message
        self.analytics.update_from_message(user_message)

        # Auto-save if enabled
        if self.config.auto_save and self.storage:
            await self.storage.save_message(self.id, user_message)

        # Only get response if we have a client and it's a user message
        if self.client and role == "user":
            response_message = await self._get_response(model, provider, **kwargs)
            return response_message

        return user_message

    async def _get_response(
        self, model: Optional[str] = None, provider: Optional[str] = None, **kwargs: Any
    ) -> ConversationMessage:
        """Get response from LLM."""
        if not self.client:
            raise ValueError("No client available for generating responses")

        # Use default model/provider if not specified
        model = model or self.config.default_model
        provider = provider or self.config.default_provider

        # Prepare messages for API call
        api_messages = await self._prepare_messages_for_api()

        # Track response time
        start_time = time.time()

        try:
            # Get response from client
            response = await self.client.completion.acreate(
                messages=api_messages, model=model, provider=provider, **kwargs
            )

            response_time = time.time() - start_time

            # Create conversation message from response
            assistant_message = ConversationMessage(
                role="assistant",
                content=response.choices[0].message.content,
                timestamp=time.time(),
                model=response.model,
                provider=response.provider,
                usage=response.usage,
                metadata={
                    "response_time": response_time,
                    "cached": getattr(response, "cached", False),
                    "raw_response_id": response.id,
                },
            )

            # Add to conversation
            self.messages.append(assistant_message)
            self.updated_at = time.time()

            # Update analytics
            self.analytics.update_from_message(assistant_message, response_time)

            # Auto-generate title if this is the first exchange
            if self.config.auto_title and not self.title and len(self.messages) >= 2:
                await self._generate_title()

            # Auto-save if enabled
            if self.config.auto_save and self.storage:
                await self.storage.save_message(self.id, assistant_message)
                await self._save_state()

            return assistant_message

        except Exception as e:
            # Track error in analytics
            self.analytics.updated_at = time.time()
            raise e

    async def _prepare_messages_for_api(self) -> List[Message]:
        """Prepare messages for API call with context management."""
        # Apply context management
        managed_messages, removed_count = self.context_manager.truncate_context(self.messages)

        if removed_count > 0:
            self.analytics.context_truncations += 1

        # Convert to API format
        api_messages = []

        # Add system prompt if configured
        if self.config.system_prompt:
            api_messages.append(Message(role=Role.SYSTEM, content=self.config.system_prompt))

        # Add conversation messages
        for msg in managed_messages:
            # Skip system messages if we already added the configured system prompt
            if msg.role == "system" and self.config.system_prompt:
                continue
            api_messages.append(msg.to_message())

        return api_messages

    async def _generate_title(self) -> None:
        """Generate a title for the conversation."""
        if not self.client or len(self.messages) < 2:
            return

        try:
            # Use first user message to generate title
            first_user_message = next((msg for msg in self.messages if msg.role == "user"), None)

            if first_user_message:
                title_prompt = f"Generate a short, descriptive title (max 6 words) for a conversation that starts with: '{first_user_message.content[:100]}...'"

                title_messages = [Message(role=Role.USER, content=title_prompt)]

                response = await self.client.completion.acreate(
                    messages=title_messages,
                    model="gpt-3.5-turbo",  # Use cheaper model for titles
                    max_tokens=20,
                    temperature=0.3,
                    _bypass_cache=True,  # Don't cache title generation
                )

                self.title = response.choices[0].message.content.strip().strip("\"'")

        except Exception:
            # If title generation fails, use fallback
            self.title = f"Conversation {self.id[:8]}"

    def add_system_message(self, content: str) -> ConversationMessage:
        """Add a system message to the conversation (synchronous)."""
        result = self._run_async(self.add_system_message_async(content))
        return result  # type: ignore[no-any-return]

    async def add_system_message_async(self, content: str) -> ConversationMessage:
        """Add a system message to the conversation (asynchronous)."""
        system_message = ConversationMessage(role="system", content=content, timestamp=time.time())

        self.messages.append(system_message)
        self.updated_at = time.time()
        self.analytics.update_from_message(system_message)

        if self.config.auto_save and self.storage:
            await self.storage.save_message(self.id, system_message)

        return system_message

    def get_history(self, limit: Optional[int] = None) -> List[ConversationMessage]:
        """Get conversation history."""
        if limit:
            return self.messages[-limit:]
        return self.messages.copy()

    def get_context_stats(self) -> Dict[str, Any]:
        """Get context window statistics."""
        return self.context_manager.get_context_stats(self.messages)

    def save(self) -> None:
        """Save the conversation to storage (synchronous)."""
        self._run_async(self.save_async())

    async def save_async(self) -> None:
        """Save the conversation to storage (asynchronous)."""
        if not self.storage:
            raise ValueError("No storage backend configured")

        await self._save_state()

    async def _save_state(self) -> None:
        """Save current conversation state."""
        if self.storage:
            await self.storage.save_conversation(
                conversation_id=self.id,
                messages=self.messages,
                summary=self.summary,
                analytics=self.analytics,
                metadata=self.metadata,
            )

    @classmethod
    def load(
        cls,
        conversation_id: str,
        storage: ConversationStorage,
        client: Optional[Any] = None,
    ) -> Optional["Conversation"]:
        """Load a conversation from storage (synchronous)."""

        async def _load() -> Optional["Conversation"]:
            return await cls.load_async(conversation_id, storage, client)

        try:
            # Try to get existing event loop
            asyncio.get_running_loop()
            # If we're already in an async context, we need to use a different approach
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _load())
                return future.result()  # type: ignore
        except RuntimeError:
            # No event loop running, we can use asyncio.run
            return asyncio.run(_load())

    @classmethod
    async def load_async(
        cls,
        conversation_id: str,
        storage: ConversationStorage,
        client: Optional[Any] = None,
    ) -> Optional["Conversation"]:
        """Load a conversation from storage (asynchronous)."""
        data = await storage.load_conversation(conversation_id)
        if not data:
            return None

        conversation = cls(conversation_id=conversation_id, storage=storage, client=client)

        # Restore state
        conversation.messages = data["messages"]
        summary = data["summary"]
        conversation.title = summary.title
        conversation.state = ConversationState(summary.state)
        conversation.created_at = summary.created_at
        conversation.updated_at = summary.updated_at
        conversation.tags = summary.tags
        conversation.metadata = summary.metadata or {}
        conversation.analytics = data["analytics"]

        return conversation

    def pause(self) -> None:
        """Pause the conversation."""
        self.state = ConversationState.PAUSED
        self.updated_at = time.time()

    def resume(self) -> None:
        """Resume the conversation."""
        self.state = ConversationState.ACTIVE
        self.updated_at = time.time()

    def complete(self) -> None:
        """Mark conversation as completed."""
        self.state = ConversationState.COMPLETED
        self.updated_at = time.time()
        self.analytics.calculate_conversation_duration(self.created_at)

    def archive(self) -> None:
        """Archive the conversation."""
        self.state = ConversationState.ARCHIVED
        self.updated_at = time.time()

    def add_tag(self, tag: str) -> None:
        """Add a tag to the conversation."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = time.time()

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the conversation."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = time.time()

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata for the conversation."""
        self.metadata[key] = value
        self.updated_at = time.time()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.metadata.get(key, default)

    def __repr__(self) -> str:
        return f"Conversation(id='{self.id}', messages={len(self.messages)}, state='{self.state.value}')"

    # Synchronous wrapper methods
    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine and return result."""
        try:
            # Try to get existing event loop
            asyncio.get_running_loop()
            # If we're already in an async context, we need to use a different approach
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()  # type: ignore
        except RuntimeError:
            # No event loop running, we can use asyncio.run
            return asyncio.run(coro)
