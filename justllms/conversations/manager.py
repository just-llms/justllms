"""Conversation manager for handling multiple conversations."""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union

from justllms.conversations.conversation import Conversation
from justllms.conversations.models import (
    ConversationConfig,
    ConversationState,
    ConversationSummary,
)
from justllms.conversations.storage import (
    ConversationStorage,
    DiskStorage,
    MemoryStorage,
    RedisStorage,
)


class ConversationManager:
    """Manages multiple conversations and provides a unified interface."""

    def __init__(
        self,
        default_config: Optional[ConversationConfig] = None,
        storage_config: Optional[Union[Dict[str, Any], Any]] = None,
        client: Optional[Any] = None,  # JustLLM client
    ):
        self.client = client
        self.default_config = default_config or ConversationConfig()

        # Store original storage config for defaults extraction
        self._storage_config = storage_config

        # Initialize storage backend
        self.storage = self._create_storage_backend(storage_config or {})

        # Active conversations cache
        self._active_conversations: Dict[str, Conversation] = {}

    def _create_storage_backend(
        self, storage_config: Union[Dict[str, Any], Any]
    ) -> ConversationStorage:
        """Create storage backend from configuration."""
        # Handle both dict and ConversationsConfig object
        if hasattr(storage_config, "model_dump"):
            # It's a Pydantic model, convert to dict
            config_dict = storage_config.model_dump()
        else:
            config_dict = storage_config or {}

        backend_type = config_dict.get("backend", "memory")
        backend_config = config_dict.get("config", {})

        if backend_type == "memory":
            return MemoryStorage()
        elif backend_type == "disk":
            return DiskStorage(**backend_config)
        elif backend_type == "redis":
            return RedisStorage(**backend_config)
        else:
            raise ValueError(f"Unknown storage backend: {backend_type}")

    async def create(
        self,
        conversation_id: Optional[str] = None,
        config: Optional[ConversationConfig] = None,
        **config_kwargs: Any,
    ) -> Conversation:
        """Create a new conversation."""
        # Merge config with defaults
        if config is None:
            config = ConversationConfig(**config_kwargs)
        elif config_kwargs:
            # Update existing config with kwargs
            config_dict = config.model_dump()
            config_dict.update(config_kwargs)
            config = ConversationConfig(**config_dict)

        # Merge with default config
        final_config_dict = self.default_config.model_dump()

        # Extract defaults from storage config if it's a ConversationsConfig
        if self._storage_config and hasattr(self._storage_config, "model_dump"):
            storage_dict = self._storage_config.model_dump()
            # Apply conversation defaults from storage config
            conversation_defaults = {
                "default_model": storage_dict.get("default_model"),
                "default_provider": storage_dict.get("default_provider"),
                "max_context_tokens": storage_dict.get("max_context_tokens", 8000),
                "context_strategy": storage_dict.get("context_strategy", "truncate"),
                "auto_save": storage_dict.get("auto_save", True),
                "auto_title": storage_dict.get("auto_title", True),
                "enable_analytics": storage_dict.get("enable_analytics", True),
            }
            # Only update with non-None values
            for key, value in conversation_defaults.items():
                if value is not None:
                    final_config_dict[key] = value

        # Apply specific config
        final_config_dict.update(config.model_dump(exclude_unset=True))
        final_config = ConversationConfig(**final_config_dict)

        conversation = Conversation(
            conversation_id=conversation_id,
            config=final_config,
            storage=self.storage,
            client=self.client,
        )

        # Cache active conversation
        self._active_conversations[conversation.id] = conversation

        # Save initial state if auto_save is enabled
        if final_config.auto_save:
            conversation.save()

        return conversation

    async def get(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        # Check active cache first
        if conversation_id in self._active_conversations:
            return self._active_conversations[conversation_id]

        # Try to load from storage
        conversation = await Conversation.load_async(
            conversation_id=conversation_id,
            storage=self.storage,
            client=self.client,
        )

        if conversation:
            # Cache the loaded conversation
            self._active_conversations[conversation_id] = conversation

        return conversation

    async def list(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        state: Optional[ConversationState] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ConversationSummary]:
        """List conversations with optional filtering."""
        filters = {}

        if state:
            filters["state"] = state
        if tags:
            filters["tags"] = tags  # type: ignore

        return await self.storage.list_conversations(limit=limit, offset=offset, filters=filters)

    async def delete(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        # Remove from active cache
        if conversation_id in self._active_conversations:
            del self._active_conversations[conversation_id]

        # Delete from storage
        return await self.storage.delete_conversation(conversation_id)

    async def search(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        state: Optional[ConversationState] = None,
        limit: Optional[int] = None,
    ) -> List[ConversationSummary]:
        """Search conversations (basic implementation)."""
        # Get all conversations with filters
        all_conversations = await self.list(
            state=state,
            tags=tags,
            limit=None,  # Get all for searching
        )

        results = all_conversations

        # Apply text search if query provided
        if query:
            query_lower = query.lower()
            results = [
                conv
                for conv in results
                if (conv.title and query_lower in conv.title.lower())
                or any(query_lower in tag.lower() for tag in conv.tags)
            ]

        # Apply limit
        if limit:
            results = results[:limit]

        return results

    def get_active_conversations(self) -> List[Conversation]:
        """Get all active (cached) conversations."""
        return list(self._active_conversations.values())

    def clear_cache(self) -> None:
        """Clear the active conversations cache."""
        self._active_conversations.clear()

    async def archive_old_conversations(
        self,
        days_threshold: int = 30,
        state_filter: Optional[ConversationState] = ConversationState.COMPLETED,
    ) -> int:
        """Archive conversations older than threshold."""
        import time

        threshold_timestamp = time.time() - (days_threshold * 24 * 60 * 60)
        archived_count = 0

        # Get conversations to archive
        conversations = await self.list(state=state_filter)

        for summary in conversations:
            if summary.updated_at < threshold_timestamp:
                conversation = await self.get(summary.id)
                if conversation:
                    conversation.archive()
                    conversation.save()
                    archived_count += 1

        return archived_count

    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary across all conversations."""
        conversations = await self.list()

        total_conversations = len(conversations)
        total_messages = sum(conv.message_count for conv in conversations)
        total_tokens = sum(conv.total_tokens for conv in conversations)
        total_cost = sum(conv.estimated_cost for conv in conversations)

        # Count by state
        state_counts: Dict[str, int] = {}
        for conv in conversations:
            state_counts[conv.state.value] = state_counts.get(conv.state.value, 0) + 1

        # Count models used
        all_models = set()
        for conv in conversations:
            all_models.update(conv.models_used)

        # Count providers used
        all_providers = set()
        for conv in conversations:
            all_providers.update(conv.providers_used)

        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "average_messages_per_conversation": (
                total_messages / total_conversations if total_conversations > 0 else 0
            ),
            "average_cost_per_conversation": (
                total_cost / total_conversations if total_conversations > 0 else 0
            ),
            "state_distribution": state_counts,
            "unique_models": len(all_models),
            "unique_providers": len(all_providers),
            "models_used": list(all_models),
            "providers_used": list(all_providers),
        }

    async def export_conversation(
        self, conversation_id: str, format: str = "json"
    ) -> Optional[Union[Dict[str, Any], str]]:
        """Export a conversation in the specified format."""
        conversation = await self.get(conversation_id)
        if not conversation:
            return None

        if format == "json":
            return {
                "id": conversation.id,
                "title": conversation.title,
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
                "state": conversation.state.value,
                "tags": conversation.tags,
                "messages": [
                    {
                        "id": msg.id,
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                        "model": msg.model,
                        "provider": msg.provider,
                        "metadata": msg.metadata,
                    }
                    for msg in conversation.messages
                ],
                "analytics": conversation.analytics.model_dump(),
                "metadata": conversation.metadata,
            }

        elif format == "markdown":
            lines = [
                f"# {conversation.title or 'Conversation ' + conversation.id[:8]}",
                "",
                f"**Created:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(conversation.created_at))}",
                f"**Updated:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(conversation.updated_at))}",
                f"**Messages:** {len(conversation.messages)}",
                f"**State:** {conversation.state.value}",
                "",
                "---",
                "",
            ]

            for msg in conversation.messages:
                timestamp = time.strftime("%H:%M:%S", time.localtime(msg.timestamp))
                lines.extend(
                    [
                        f"## {msg.role.title()} [{timestamp}]",
                        "",
                        msg.content,
                        "",
                    ]
                )

            return "\n".join(lines)

        elif format == "txt":
            lines = [
                f"{conversation.title or 'Conversation ' + conversation.id[:8]}",
                "=" * 50,
                "",
            ]

            for msg in conversation.messages:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(msg.timestamp))
                lines.extend(
                    [
                        f"{msg.role.upper()} [{timestamp}]:",
                        msg.content,
                        "",
                    ]
                )

            return "\n".join(lines)

        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def import_conversation(
        self, data: Dict[str, Any], conversation_id: Optional[str] = None
    ) -> Conversation:
        """Import a conversation from exported data."""
        # Create new conversation
        conversation = await self.create(conversation_id=conversation_id or data.get("id"))

        # Import basic data
        if "title" in data:
            conversation.title = data["title"]
        if "tags" in data:
            conversation.tags = data["tags"]
        if "metadata" in data:
            conversation.metadata = data["metadata"]
        if "created_at" in data:
            conversation.created_at = data["created_at"]
        if "state" in data:
            conversation.state = ConversationState(data["state"])

        # Import messages
        if "messages" in data:
            from justllms.conversations.models import ConversationMessage

            for msg_data in data["messages"]:
                message = ConversationMessage(**msg_data)
                conversation.messages.append(message)
                conversation.analytics.update_from_message(message)

        conversation.updated_at = time.time()

        # Save the imported conversation
        conversation.save()

        return conversation

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
                return future.result()
        except RuntimeError:
            # No event loop running, we can use asyncio.run
            return asyncio.run(coro)

    def create_sync(
        self,
        conversation_id: Optional[str] = None,
        config: Optional[ConversationConfig] = None,
        **config_kwargs: Any,
    ) -> Conversation:
        """Create a new conversation (synchronous)."""
        return self._run_async(self.create(conversation_id, config, **config_kwargs))  # type: ignore

    def get_sync(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID (synchronous)."""
        return self._run_async(self.get(conversation_id))  # type: ignore

    def list_sync(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        state: Optional[ConversationState] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ConversationSummary]:
        """List conversations with optional filtering (synchronous)."""
        return self._run_async(self.list(limit, offset, state, tags))  # type: ignore

    def delete_sync(self, conversation_id: str) -> bool:
        """Delete a conversation (synchronous)."""
        return self._run_async(self.delete(conversation_id))  # type: ignore

    def search_sync(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        state: Optional[ConversationState] = None,
        limit: Optional[int] = None,
    ) -> List[ConversationSummary]:
        """Search conversations (synchronous)."""
        return self._run_async(self.search(query, tags, state, limit))  # type: ignore

    def archive_old_conversations_sync(
        self,
        days_threshold: int = 30,
        state_filter: Optional[ConversationState] = ConversationState.COMPLETED,
    ) -> int:
        """Archive conversations older than threshold (synchronous)."""
        return self._run_async(self.archive_old_conversations(days_threshold, state_filter))  # type: ignore

    def get_analytics_summary_sync(self) -> Dict[str, Any]:
        """Get analytics summary across all conversations (synchronous)."""
        return self._run_async(self.get_analytics_summary())  # type: ignore

    def export_conversation_sync(
        self, conversation_id: str, format: str = "json"
    ) -> Optional[Union[Dict[str, Any], str]]:
        """Export a conversation in the specified format (synchronous)."""
        return self._run_async(self.export_conversation(conversation_id, format))  # type: ignore

    def import_conversation_sync(
        self, data: Dict[str, Any], conversation_id: Optional[str] = None
    ) -> Conversation:
        """Import a conversation from exported data (synchronous)."""
        return self._run_async(self.import_conversation(data, conversation_id))  # type: ignore
