"""Context window management for conversations."""

from typing import Dict, List, Tuple

import tiktoken

from justllms.conversations.models import ConversationConfig, ConversationMessage


class ContextManager:
    """Manages conversation context and token limits."""

    def __init__(self, config: ConversationConfig):
        self.config = config
        self.encoder = None

        # Initialize tokenizer
        try:
            # Use cl100k_base encoding (GPT-4/GPT-3.5 compatible)
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback to approximate counting
            self.encoder = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4

    def count_message_tokens(self, message: ConversationMessage) -> int:
        """Count tokens in a message."""
        # Add overhead for role and formatting
        base_tokens = 4  # Overhead for message structure
        content_tokens = self.count_tokens(message.content)
        role_tokens = self.count_tokens(message.role)

        return base_tokens + content_tokens + role_tokens

    def count_conversation_tokens(self, messages: List[ConversationMessage]) -> int:
        """Count total tokens in conversation."""
        total = 0
        for message in messages:
            total += self.count_message_tokens(message)

        # Add overhead for conversation structure
        total += 3  # Base conversation overhead

        return total

    def should_truncate_context(self, messages: List[ConversationMessage]) -> bool:
        """Check if context should be truncated."""
        if not self.config.max_context_tokens:
            return False

        total_tokens = self.count_conversation_tokens(messages)
        return total_tokens > self.config.max_context_tokens

    def truncate_context(
        self, messages: List[ConversationMessage]
    ) -> Tuple[List[ConversationMessage], int]:
        """Truncate context based on strategy."""
        if not self.should_truncate_context(messages):
            return messages, 0

        strategy = self.config.context_strategy

        if strategy == "truncate":
            return self._truncate_oldest(messages)
        elif strategy == "summarize":
            return self._summarize_old_messages(messages)
        elif strategy == "compress":
            return self._compress_messages(messages)
        else:
            # Default to truncation
            return self._truncate_oldest(messages)

    def _truncate_oldest(
        self, messages: List[ConversationMessage]
    ) -> Tuple[List[ConversationMessage], int]:
        """Truncate oldest messages to fit context window."""
        if not self.config.max_context_tokens:
            return messages, 0

        # Always keep system prompt if configured
        system_messages = []
        other_messages = []

        for msg in messages:
            if msg.role == "system" and self.config.keep_system_prompt:
                system_messages.append(msg)
            else:
                other_messages.append(msg)

        # Start with system messages
        result = system_messages.copy()
        removed_count = 0

        # Add messages from newest to oldest until we hit the limit
        for message in reversed(other_messages):
            temp_result = [message] + result
            if self.count_conversation_tokens(temp_result) <= self.config.max_context_tokens:
                result.insert(len(system_messages), message)
            else:
                removed_count += 1

        # Ensure chronological order
        if system_messages:
            non_system = [msg for msg in result if msg.role != "system"]
            result = system_messages + non_system

        return result, removed_count

    def _summarize_old_messages(
        self, messages: List[ConversationMessage]
    ) -> Tuple[List[ConversationMessage], int]:
        """Summarize old messages to save context space."""
        # This is a simplified implementation
        # In a real implementation, you'd use an LLM to generate summaries

        if not self.config.max_context_tokens:
            return messages, 0

        # Keep system messages and recent messages
        system_messages = [msg for msg in messages if msg.role == "system"]
        other_messages = [msg for msg in messages if msg.role != "system"]

        if len(other_messages) <= 4:  # Keep if conversation is short
            return messages, 0

        # Keep last N messages and summarize the rest
        keep_recent = 4
        recent_messages = other_messages[-keep_recent:]
        old_messages = other_messages[:-keep_recent]

        # Create a summary message (placeholder implementation)
        if old_messages:
            summary_content = f"[Previous conversation summary: {len(old_messages)} messages about various topics]"
            summary_message = ConversationMessage(
                role="system",
                content=summary_content,
                metadata={"type": "summary", "original_message_count": len(old_messages)},
            )

            result = system_messages + [summary_message] + recent_messages
            return result, len(old_messages)

        return messages, 0

    def _compress_messages(
        self, messages: List[ConversationMessage]
    ) -> Tuple[List[ConversationMessage], int]:
        """Compress messages to fit context window."""
        if not self.config.max_context_tokens:
            return messages, 0

        target_tokens = int(self.config.max_context_tokens * self.config.context_compression_ratio)
        current_tokens = self.count_conversation_tokens(messages)

        if current_tokens <= target_tokens:
            return messages, 0

        # Simple compression: remove every other message starting from oldest
        # Keep system messages and most recent messages
        system_messages = [msg for msg in messages if msg.role == "system"]
        other_messages = [msg for msg in messages if msg.role != "system"]

        if len(other_messages) <= 2:
            return messages, 0

        # Keep recent messages, compress middle ones
        keep_recent = 2
        recent_messages = other_messages[-keep_recent:]
        middle_messages = other_messages[:-keep_recent]

        # Compress by taking every other message
        compressed_messages = []
        removed_count = 0

        for i, msg in enumerate(middle_messages):
            if i % 2 == 0:  # Keep every other message
                compressed_messages.append(msg)
            else:
                removed_count += 1

        result = system_messages + compressed_messages + recent_messages

        # Check if we're still over the limit
        if self.count_conversation_tokens(result) > target_tokens:
            # Fall back to truncation
            return self._truncate_oldest(result)

        return result, removed_count

    def optimize_for_model(
        self, messages: List[ConversationMessage], model: str
    ) -> List[ConversationMessage]:
        """Optimize context for specific model constraints."""
        # Model-specific optimizations could be added here
        # For now, just apply general truncation
        optimized_messages, _ = self.truncate_context(messages)
        return optimized_messages

    def get_context_stats(self, messages: List[ConversationMessage]) -> dict:
        """Get statistics about the current context."""
        total_tokens = self.count_conversation_tokens(messages)

        role_counts: Dict[str, int] = {}
        role_tokens: Dict[str, int] = {}

        for message in messages:
            role = message.role
            role_counts[role] = role_counts.get(role, 0) + 1
            role_tokens[role] = role_tokens.get(role, 0) + self.count_message_tokens(message)

        return {
            "total_messages": len(messages),
            "total_tokens": total_tokens,
            "max_tokens": self.config.max_context_tokens,
            "utilization": (
                total_tokens / self.config.max_context_tokens
                if self.config.max_context_tokens
                else 0
            ),
            "needs_truncation": self.should_truncate_context(messages),
            "role_counts": role_counts,
            "role_tokens": role_tokens,
            "strategy": self.config.context_strategy,
        }
