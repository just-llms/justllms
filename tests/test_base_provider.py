"""Tests for BaseProvider shared helpers."""

from justllms.core.base import BaseProvider
from justllms.core.models import ProviderConfig, Role


class MinimalProvider(BaseProvider):
    """Concrete provider exposing protected helpers for unit tests."""

    requires_api_key = False

    @property
    def name(self) -> str:
        return "minimal"

    def complete(self, messages, model, timeout=None, **kwargs):
        raise NotImplementedError

    def get_available_models(self):
        return {}


class TestCreateStandardChoice:
    def test_preserves_tool_call_id_on_tool_messages(self) -> None:
        provider = MinimalProvider(ProviderConfig(name="minimal"))

        choice = provider._create_standard_choice(
            {
                "role": Role.TOOL,
                "content": '{"temperature": 22}',
                "name": "get_weather",
                "tool_call_id": "call_abc123",
            }
        )

        assert choice.message.role == Role.TOOL
        assert choice.message.tool_call_id == "call_abc123"
        assert choice.message.name == "get_weather"

    def test_tool_call_id_absent_when_not_in_payload(self) -> None:
        provider = MinimalProvider(ProviderConfig(name="minimal"))

        choice = provider._create_standard_choice(
            {
                "role": Role.ASSISTANT,
                "content": "Hello",
            }
        )

        assert choice.message.tool_call_id is None
