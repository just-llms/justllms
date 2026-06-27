"""Tests for Anthropic provider tool calling and streaming."""

from justllms.core.models import Message, ProviderConfig, Role
from justllms.providers.anthropic import AnthropicProvider


def _provider():
    return AnthropicProvider(ProviderConfig(name="anthropic", api_key="test-key"))


def test_build_payload_maps_params():
    provider = _provider()
    messages = [
        Message(role=Role.SYSTEM, content="be brief"),
        Message(role=Role.USER, content="hi"),
    ]

    payload = provider._build_payload(
        messages,
        "claude-opus-4-7",
        max_tokens=256,
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        stop="STOP",
    )

    assert payload["model"] == "claude-opus-4-7"
    assert payload["max_tokens"] == 256
    assert payload["system"] == "be brief"
    assert payload["temperature"] == 0.5
    assert payload["top_p"] == 0.9
    assert payload["top_k"] == 40
    assert payload["stop_sequences"] == ["STOP"]
    # System message is not part of the messages array.
    assert all(m["role"] != "system" for m in payload["messages"])


def test_complete_with_tools_sends_tools(monkeypatch):
    provider = _provider()
    captured = {}

    def fake_request(url, payload, headers, timeout=None, **kwargs):
        captured["payload"] = payload
        return {
            "id": "msg_1",
            "model": "claude-opus-4-7",
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"location": "Paris"},
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    monkeypatch.setattr(provider, "_make_http_request", fake_request)

    tools = [
        {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]

    response = provider.complete_with_tools(
        messages=[Message(role=Role.USER, content="weather in Paris?")],
        tools=tools,
        model="claude-opus-4-7",
        tool_choice={"type": "auto"},
        max_tokens=100,
    )

    # Tools and tool_choice must reach the payload (the original bug dropped them).
    assert captured["payload"]["tools"] == tools
    assert captured["payload"]["tool_choice"] == {"type": "auto"}

    # The raw tool_use content must survive for the adapter to extract.
    adapter = provider.get_tool_adapter()
    tool_calls = adapter.extract_tool_calls(dict(response.raw_response))
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert tool_calls[0].arguments == {"location": "Paris"}


def test_complete_with_tools_omits_tool_choice_without_tools(monkeypatch):
    provider = _provider()
    captured = {}

    def fake_request(url, payload, headers, timeout=None, **kwargs):
        captured["payload"] = payload
        return {
            "id": "msg_1",
            "model": "claude-opus-4-7",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(provider, "_make_http_request", fake_request)

    provider.complete_with_tools(
        messages=[Message(role=Role.USER, content="hi")],
        tools=None,
        model="claude-opus-4-7",
        tool_choice={"type": "auto"},
    )

    assert "tools" not in captured["payload"]
    assert "tool_choice" not in captured["payload"]


SSE_TEXT_STREAM = [
    "event: message_start",
    'data: {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 25, "output_tokens": 1}}}',
    "",
    "event: content_block_start",
    'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}',
    "",
    "event: ping",
    'data: {"type": "ping"}',
    "",
    "event: content_block_delta",
    'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}',
    "",
    "event: content_block_delta",
    'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}}',
    "",
    "event: content_block_stop",
    'data: {"type": "content_block_stop", "index": 0}',
    "",
    "event: message_delta",
    'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 15}}',
    "",
    "event: message_stop",
    'data: {"type": "message_stop"}',
]


def test_iter_anthropic_chunks_text_and_usage():
    provider = _provider()
    chunks = list(provider._iter_anthropic_chunks(SSE_TEXT_STREAM))

    text = "".join(c.content for c in chunks if c.content)
    assert text == "Hello world"

    final = [c for c in chunks if c.finish_reason is not None]
    assert len(final) == 1
    assert final[0].finish_reason == "end_turn"
    # input_tokens come from message_start, output_tokens from message_delta.
    assert final[0].usage.prompt_tokens == 25
    assert final[0].usage.completion_tokens == 15
    assert final[0].usage.total_tokens == 40


def test_iter_anthropic_chunks_ignores_non_data_and_bad_json():
    provider = _provider()
    lines = [
        "event: message_start",
        "data: not-json",
        ": comment",
        'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}}',
    ]
    chunks = list(provider._iter_anthropic_chunks(lines))
    assert [c.content for c in chunks] == ["ok"]


def test_supports_streaming():
    provider = _provider()
    assert provider.supports_streaming() is True
    assert provider.supports_streaming_for_model("claude-opus-4-7") is True
    assert provider.supports_streaming_for_model("not-a-model") is False
