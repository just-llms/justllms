"""Tests for message validation."""

import pytest

from justllms.core.models import Message, Role
from justllms.exceptions import ValidationError
from justllms.utils.validators import validate_messages


class TestValidateMessages:
    def test_dict_messages_validated(self):
        messages = validate_messages([{"role": "user", "content": "Hello"}])
        assert len(messages) == 1
        assert messages[0].role == Role.USER

    def test_message_objects_validated(self):
        messages = validate_messages([Message(role=Role.USER, content="Hello")])
        assert len(messages) == 1
        assert messages[0].content == "Hello"

    def test_message_object_empty_content_rejected(self):
        with pytest.raises(ValidationError, match="content cannot be empty"):
            validate_messages([Message(role=Role.USER, content="")])

    def test_message_object_whitespace_content_rejected(self):
        with pytest.raises(ValidationError, match="content cannot be empty"):
            validate_messages([Message(role=Role.USER, content="   ")])

    def test_message_object_tool_role_allows_empty_content(self):
        messages = validate_messages(
            [
                Message(role=Role.USER, content="What's the weather?"),
                Message(role=Role.TOOL, content="", tool_call_id="call_123"),
            ]
        )
        assert messages[1].role == Role.TOOL

    def test_message_object_assistant_with_tool_calls_allows_empty_content(self):
        messages = validate_messages(
            [
                Message(role=Role.USER, content="What's the weather?"),
                Message(
                    role=Role.ASSISTANT,
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                ),
            ]
        )
        assert messages[1].tool_calls is not None

    def test_dict_and_message_mixed_input(self):
        messages = validate_messages(
            [
                Message(role=Role.USER, content="Hi"),
                {"role": "assistant", "content": "Hello"},
            ]
        )
        assert len(messages) == 2

    def test_empty_messages_list_rejected(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_messages([])
