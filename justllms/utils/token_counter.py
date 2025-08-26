from typing import Any, Dict, List, Optional, Union

try:
    import tiktoken

    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


class TokenCounter:
    """Count tokens for different models."""

    # Model to encoding mapping
    MODEL_ENCODINGS = {
        "gpt-4": "cl100k_base",
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
        "claude": "cl100k_base",  # Approximation
        "gemini": "cl100k_base",  # Approximation
    }

    # Rough token estimates when tiktoken is not available
    CHARS_PER_TOKEN = {
        "default": 4,
        "chinese": 2,
        "japanese": 2,
        "korean": 2,
    }

    def __init__(self) -> None:
        self._encodings: Dict[str, Any] = {}

    def _get_encoding(self, model: str) -> Optional[Any]:
        """Get the encoding for a model."""
        if not HAS_TIKTOKEN:
            return None

        # Find the encoding name for the model
        encoding_name = None

        # Check exact match first
        if model in self.MODEL_ENCODINGS:
            encoding_name = self.MODEL_ENCODINGS[model]
        else:
            # Check prefixes
            for model_prefix, enc_name in self.MODEL_ENCODINGS.items():
                if model.startswith(model_prefix):
                    encoding_name = enc_name
                    break

        if not encoding_name:
            encoding_name = "cl100k_base"  # Default

        # Cache encodings
        if encoding_name not in self._encodings:
            try:
                self._encodings[encoding_name] = tiktoken.get_encoding(encoding_name)
            except Exception:
                return None

        return self._encodings[encoding_name]

    def count_tokens(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> int:
        """Count tokens in text."""
        if HAS_TIKTOKEN and model:
            encoding = self._get_encoding(model)
            if encoding:
                try:
                    return len(encoding.encode(text))
                except Exception:
                    pass

        # Fallback to character-based estimation
        return self._estimate_tokens(text)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count based on character count."""
        # Simple heuristic: ~4 characters per token for English
        # Adjust for other languages if detected

        # Check for CJK characters
        cjk_count = sum(
            1
            for char in text
            if "\u4e00" <= char <= "\u9fff"  # Chinese
            or "\u3040" <= char <= "\u309f"  # Hiragana
            or "\u30a0" <= char <= "\u30ff"  # Katakana
            or "\uac00" <= char <= "\ud7af"  # Korean
        )

        chars_per_token = 2 if cjk_count > len(text) * 0.3 else 4

        return max(1, len(text) // chars_per_token)

    def count_messages_tokens(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> Dict[str, int]:
        """Count tokens in a list of messages."""
        total_tokens = 0
        per_message_tokens = 4  # Overhead per message

        for message in messages:
            # Count role tokens
            role = message.get("role", "")
            total_tokens += self.count_tokens(role, model)

            # Count content tokens
            content = message.get("content", "")
            if isinstance(content, str):
                total_tokens += self.count_tokens(content, model)
            elif isinstance(content, list):
                # Handle multimodal content
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total_tokens += self.count_tokens(item.get("text", ""), model)
                    elif isinstance(item, dict) and item.get("type") == "image":
                        # Rough estimate for images
                        total_tokens += 85  # Base64 encoded image token estimate

            # Add per-message overhead
            total_tokens += per_message_tokens

            # Handle other fields
            if message.get("name"):
                total_tokens += self.count_tokens(message["name"], model)

            if message.get("function_call"):
                total_tokens += self.count_tokens(str(message["function_call"]), model)

        # Add base prompt tokens
        total_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>

        return {
            "total": total_tokens,
            "messages": len(messages),
        }


# Global instance
_token_counter = TokenCounter()


def count_tokens(
    text: Union[str, List[Dict[str, Any]]],
    model: Optional[str] = None,
) -> int:
    """Count tokens in text or messages."""
    if isinstance(text, str):
        return _token_counter.count_tokens(text, model)
    elif isinstance(text, list):
        return _token_counter.count_messages_tokens(text, model)["total"]
    else:
        return 0
