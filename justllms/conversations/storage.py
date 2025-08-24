"""Storage backends for conversation persistence."""

import asyncio
import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from justllms.conversations.models import (
    ConversationAnalytics,
    ConversationMessage,
    ConversationSummary,
)


class ConversationStorage(ABC):
    """Abstract base class for conversation storage backends."""

    @abstractmethod
    async def save_conversation(
        self,
        conversation_id: str,
        messages: List[ConversationMessage],
        summary: ConversationSummary,
        analytics: ConversationAnalytics,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a complete conversation."""
        pass

    @abstractmethod
    async def load_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Load a conversation by ID."""
        pass

    @abstractmethod
    async def list_conversations(
        self, limit: Optional[int] = None, offset: int = 0, filters: Optional[Dict[str, Any]] = None
    ) -> List[ConversationSummary]:
        """List conversations with optional filtering."""
        pass

    @abstractmethod
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        pass

    @abstractmethod
    async def save_message(self, conversation_id: str, message: ConversationMessage) -> None:
        """Save a single message to a conversation."""
        pass

    @abstractmethod
    async def get_messages(
        self, conversation_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> List[ConversationMessage]:
        """Get messages from a conversation."""
        pass


class MemoryStorage(ConversationStorage):
    """In-memory storage for conversations."""

    def __init__(self) -> None:
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def save_conversation(
        self,
        conversation_id: str,
        messages: List[ConversationMessage],
        summary: ConversationSummary,
        analytics: ConversationAnalytics,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a complete conversation."""
        async with self._lock:
            self.conversations[conversation_id] = {
                "messages": [msg.model_dump() for msg in messages],
                "summary": summary.model_dump(),
                "analytics": analytics.model_dump(),
                "metadata": metadata or {},
            }

    async def load_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Load a conversation by ID."""
        async with self._lock:
            data = self.conversations.get(conversation_id)
            if not data:
                return None

            # Reconstruct objects
            messages = [ConversationMessage(**msg) for msg in data["messages"]]
            summary = ConversationSummary(**data["summary"])
            analytics = ConversationAnalytics(**data["analytics"])

            return {
                "messages": messages,
                "summary": summary,
                "analytics": analytics,
                "metadata": data.get("metadata", {}),
            }

    async def list_conversations(
        self, limit: Optional[int] = None, offset: int = 0, filters: Optional[Dict[str, Any]] = None
    ) -> List[ConversationSummary]:
        """List conversations with optional filtering."""
        async with self._lock:
            summaries = []
            for conv_data in self.conversations.values():
                summary = ConversationSummary(**conv_data["summary"])

                # Apply filters if provided
                if filters:
                    if filters.get("state") and summary.state != filters["state"]:
                        continue
                    if filters.get("tags") and not any(
                        tag in summary.tags for tag in filters["tags"]
                    ):
                        continue

                summaries.append(summary)

            # Sort by updated_at descending
            summaries.sort(key=lambda x: x.updated_at, reverse=True)

            # Apply pagination
            if limit:
                summaries = summaries[offset : offset + limit]

            return summaries

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        async with self._lock:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
                return True
            return False

    async def save_message(self, conversation_id: str, message: ConversationMessage) -> None:
        """Save a single message to a conversation."""
        async with self._lock:
            if conversation_id in self.conversations:
                self.conversations[conversation_id]["messages"].append(message.model_dump())

    async def get_messages(
        self, conversation_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> List[ConversationMessage]:
        """Get messages from a conversation."""
        async with self._lock:
            if conversation_id not in self.conversations:
                return []

            messages_data = self.conversations[conversation_id]["messages"]
            messages = [ConversationMessage(**msg) for msg in messages_data]

            if limit:
                messages = messages[offset : offset + limit]

            return messages


class DiskStorage(ConversationStorage):
    """Disk-based storage for conversations."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path.home() / ".justllms" / "conversations"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _get_conversation_path(self, conversation_id: str) -> Path:
        """Get file path for a conversation."""
        return self.storage_dir / f"{conversation_id}.json"

    async def save_conversation(
        self,
        conversation_id: str,
        messages: List[ConversationMessage],
        summary: ConversationSummary,
        analytics: ConversationAnalytics,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a complete conversation."""
        data = {
            "messages": [msg.model_dump() for msg in messages],
            "summary": summary.model_dump(),
            "analytics": analytics.model_dump(),
            "metadata": metadata or {},
        }

        async with self._lock:
            file_path = self._get_conversation_path(conversation_id)
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: file_path.write_text(json.dumps(data, indent=2))
            )

    async def load_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Load a conversation by ID."""
        file_path = self._get_conversation_path(conversation_id)

        if not file_path.exists():
            return None

        async with self._lock:
            try:
                data = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: json.loads(file_path.read_text())
                )

                # Reconstruct objects
                messages = [ConversationMessage(**msg) for msg in data["messages"]]
                summary = ConversationSummary(**data["summary"])
                analytics = ConversationAnalytics(**data["analytics"])

                return {
                    "messages": messages,
                    "summary": summary,
                    "analytics": analytics,
                    "metadata": data.get("metadata", {}),
                }
            except (json.JSONDecodeError, KeyError):
                # Error loading conversation - may not exist
                return None

    async def list_conversations(
        self, limit: Optional[int] = None, offset: int = 0, filters: Optional[Dict[str, Any]] = None
    ) -> List[ConversationSummary]:
        """List conversations with optional filtering."""
        summaries = []

        async with self._lock:
            for file_path in self.storage_dir.glob("*.json"):
                try:
                    data = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda fp=file_path: json.loads(fp.read_text()),  # type: ignore
                    )

                    summary = ConversationSummary(**data["summary"])

                    # Apply filters if provided
                    if filters:
                        if filters.get("state") and summary.state != filters["state"]:
                            continue
                        if filters.get("tags") and not any(
                            tag in summary.tags for tag in filters["tags"]
                        ):
                            continue

                    summaries.append(summary)

                except (json.JSONDecodeError, KeyError):
                    continue

        # Sort by updated_at descending
        summaries.sort(key=lambda x: x.updated_at, reverse=True)

        # Apply pagination
        if limit:
            summaries = summaries[offset : offset + limit]

        return summaries

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        file_path = self._get_conversation_path(conversation_id)

        async with self._lock:
            try:
                await asyncio.get_event_loop().run_in_executor(None, file_path.unlink)
                return True
            except FileNotFoundError:
                return False

    async def save_message(self, conversation_id: str, message: ConversationMessage) -> None:
        """Save a single message to a conversation."""
        # For disk storage, we need to load, modify, and save the entire conversation
        conversation_data = await self.load_conversation(conversation_id)
        if conversation_data:
            conversation_data["messages"].append(message)
            await self.save_conversation(
                conversation_id,
                conversation_data["messages"],
                conversation_data["summary"],
                conversation_data["analytics"],
                conversation_data["metadata"],
            )

    async def get_messages(
        self, conversation_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> List[ConversationMessage]:
        """Get messages from a conversation."""
        conversation_data = await self.load_conversation(conversation_id)
        if not conversation_data:
            return []

        messages = conversation_data["messages"]

        if limit:
            messages = messages[offset : offset + limit]

        return messages  # type: ignore  # type: ignore


class RedisStorage(ConversationStorage):
    """Redis-based storage for conversations."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "justllms:conversations:",
        **redis_kwargs: Any,
    ) -> None:
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.prefix = prefix
        self.redis_kwargs = redis_kwargs

        try:
            import redis.asyncio as aioredis

            self.redis_available = True

            self.redis = aioredis.Redis(
                host=host, port=port, db=db, password=password, **redis_kwargs
            )
        except ImportError as e:
            self.redis_available = False
            raise ImportError(
                "Redis support requires the 'redis' package. Install with: pip install redis"
            ) from e

    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.prefix}{key}"

    def _serialize(self, data: Any) -> bytes:
        """Serialize data for Redis storage."""
        return pickle.dumps(data)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from Redis storage."""
        return pickle.loads(data)

    async def save_conversation(
        self,
        conversation_id: str,
        messages: List[ConversationMessage],
        summary: ConversationSummary,
        analytics: ConversationAnalytics,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a complete conversation."""
        data = {
            "messages": [msg.model_dump() for msg in messages],
            "summary": summary.model_dump(),
            "analytics": analytics.model_dump(),
            "metadata": metadata or {},
        }

        key = self._make_key(conversation_id)
        serialized_data = self._serialize(data)

        await self.redis.set(key, serialized_data)

    async def load_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Load a conversation by ID."""
        key = self._make_key(conversation_id)
        data = await self.redis.get(key)

        if not data:
            return None

        try:
            deserialized_data = self._deserialize(data)

            # Reconstruct objects
            messages = [ConversationMessage(**msg) for msg in deserialized_data["messages"]]
            summary = ConversationSummary(**deserialized_data["summary"])
            analytics = ConversationAnalytics(**deserialized_data["analytics"])

            return {
                "messages": messages,
                "summary": summary,
                "analytics": analytics,
                "metadata": deserialized_data.get("metadata", {}),
            }
        except Exception:
            # Error deserializing conversation - corrupted data
            return None

    async def list_conversations(
        self, limit: Optional[int] = None, offset: int = 0, filters: Optional[Dict[str, Any]] = None
    ) -> List[ConversationSummary]:
        """List conversations with optional filtering."""
        pattern = f"{self.prefix}*"
        keys = []

        async for key in self.redis.scan_iter(match=pattern):
            keys.append(key.decode("utf-8"))

        summaries = []

        for key in keys:
            data = await self.redis.get(key)
            if data:
                try:
                    deserialized_data = self._deserialize(data)
                    summary = ConversationSummary(**deserialized_data["summary"])

                    # Apply filters if provided
                    if filters:
                        if filters.get("state") and summary.state != filters["state"]:
                            continue
                        if filters.get("tags") and not any(
                            tag in summary.tags for tag in filters["tags"]
                        ):
                            continue

                    summaries.append(summary)
                except Exception:
                    continue

        # Sort by updated_at descending
        summaries.sort(key=lambda x: x.updated_at, reverse=True)

        # Apply pagination
        if limit:
            summaries = summaries[offset : offset + limit]

        return summaries

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        key = self._make_key(conversation_id)
        result = await self.redis.delete(key)
        return result > 0  # type: ignore

    async def save_message(self, conversation_id: str, message: ConversationMessage) -> None:
        """Save a single message to a conversation."""
        # For Redis, we need to load, modify, and save the entire conversation
        conversation_data = await self.load_conversation(conversation_id)
        if conversation_data:
            conversation_data["messages"].append(message)
            await self.save_conversation(
                conversation_id,
                conversation_data["messages"],
                conversation_data["summary"],
                conversation_data["analytics"],
                conversation_data["metadata"],
            )

    async def get_messages(
        self, conversation_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> List[ConversationMessage]:
        """Get messages from a conversation."""
        conversation_data = await self.load_conversation(conversation_id)
        if not conversation_data:
            return []

        messages = conversation_data["messages"]

        if limit:
            messages = messages[offset : offset + limit]

        return messages  # type: ignore
