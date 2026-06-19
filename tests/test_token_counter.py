"""Tests for token counter encoding selection."""

import pytest

from justllms.core.models import Message, Role
from justllms.utils.token_counter import TokenCounter, count_tokens


def test_encoding_uses_longest_matching_prefix() -> None:
    """Versioned model IDs should match the most specific prefix."""
    counter = TokenCounter()

    gpt4o_encoding = counter._get_encoding("gpt-4o-2024-08-06")
    if gpt4o_encoding is None:
        pytest.skip("tiktoken encodings unavailable (offline environment)")

    gpt4_encoding = counter._get_encoding("gpt-4-turbo-2024-04-09")
    gpt5_encoding = counter._get_encoding("gpt-5-mini-2025-01-01")

    assert gpt4_encoding is not None
    assert gpt5_encoding is not None

    assert gpt4o_encoding.name == "o200k_base"
    assert gpt4_encoding.name == "cl100k_base"
    assert gpt5_encoding.name == "o200k_base"


class TestCountMessagesTokens:
    def test_message_objects_supported(self) -> None:
        """count_tokens must accept Message objects, not only dicts."""
        messages = [Message(role=Role.USER, content="hello world")]
        dict_count = count_tokens([{"role": "user", "content": "hello world"}], "gpt-4o")
        message_count = count_tokens(messages, "gpt-4o")
        assert message_count == dict_count
        assert message_count > 0

    def test_mixed_message_and_dict_input(self) -> None:
        messages = [
            Message(role=Role.USER, content="hello"),
            {"role": "assistant", "content": "hi there"},
        ]
        mixed_count = count_tokens(messages, "gpt-4o")
        dict_count = count_tokens(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
            "gpt-4o",
        )
        assert mixed_count == dict_count

    def test_tool_fields_included_in_count(self) -> None:
        counter = TokenCounter()
        without_tools = counter.count_messages_tokens(
            [{"role": "user", "content": "What's the weather?"}],
            "gpt-4o",
        )["total"]
        with_tools = counter.count_messages_tokens(
            [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "content": '{"temp": 22}', "tool_call_id": "call_1"},
            ],
            "gpt-4o",
        )["total"]
        assert with_tools > without_tools
