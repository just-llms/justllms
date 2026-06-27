"""Tests for tool calling on Grok, DeepSeek, and Ollama providers."""

from justllms.core.models import Message, ProviderConfig, Role
from justllms.providers.deepseek import DeepSeekProvider
from justllms.providers.grok import GrokProvider
from justllms.providers.ollama import OllamaProvider
from justllms.tools.adapters.ollama import OllamaToolAdapter
from justllms.tools.adapters.openai import OpenAIToolAdapter
from justllms.tools.models import ToolCall, ToolResult, ToolResultStatus

# --------------------------------------------------------------------------- #
# Provider capability wiring
# --------------------------------------------------------------------------- #


def test_providers_advertise_tool_support():
    for provider_cls, cfg in (
        (DeepSeekProvider, ProviderConfig(name="deepseek", api_key="k")),
        (GrokProvider, ProviderConfig(name="grok", api_key="k")),
        (OllamaProvider, ProviderConfig(name="ollama")),
    ):
        provider = provider_cls(cfg)
        assert provider.supports_tools is True
        assert provider.get_tool_adapter() is not None


def test_deepseek_and_grok_use_openai_adapter():
    deepseek = DeepSeekProvider(ProviderConfig(name="deepseek", api_key="k"))
    grok = GrokProvider(ProviderConfig(name="grok", api_key="k"))
    assert isinstance(deepseek.get_tool_adapter(), OpenAIToolAdapter)
    assert isinstance(grok.get_tool_adapter(), OpenAIToolAdapter)


def test_grok_supports_streaming():
    grok = GrokProvider(ProviderConfig(name="grok", api_key="k"))
    assert grok.supports_streaming() is True
    assert grok.supports_streaming_for_model("grok-4") is True


# --------------------------------------------------------------------------- #
# Grok: OpenAI-compatible payload + image conversion
# --------------------------------------------------------------------------- #


def test_grok_build_payload_includes_tools(monkeypatch):
    grok = GrokProvider(ProviderConfig(name="grok", api_key="k"))
    captured = {}

    def fake_request(url, payload, headers, timeout=None, **kwargs):
        captured["payload"] = payload
        captured["url"] = url
        return {
            "id": "x",
            "model": "grok-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Paris"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(grok, "_make_http_request", fake_request)

    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    response = grok.complete_with_tools(
        messages=[Message(role=Role.USER, content="weather?")],
        tools=tools,
        model="grok-4",
        tool_choice="auto",
    )

    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["payload"]["tools"] == tools
    assert captured["payload"]["tool_choice"] == "auto"

    calls = grok.get_tool_adapter().extract_tool_calls(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": response.choices[0].message.tool_calls,
                    }
                }
            ]
        }
    )
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == {"location": "Paris"}


def test_grok_converts_image_content():
    grok = GrokProvider(ProviderConfig(name="grok", api_key="k"))
    messages = [
        Message(
            role=Role.USER,
            content=[
                {"type": "text", "text": "describe"},
                {"type": "image", "image": {"url": "http://x/y.png"}},
            ],
        )
    ]

    formatted = grok._format_messages_openai(messages)
    parts = formatted[0]["content"]
    assert parts[0] == {"type": "text", "text": "describe"}
    assert parts[1] == {"type": "image_url", "image_url": {"url": "http://x/y.png"}}


# --------------------------------------------------------------------------- #
# Ollama adapter: object arguments, synthesized id, tool_name results
# --------------------------------------------------------------------------- #


def test_ollama_adapter_extracts_object_arguments():
    adapter = OllamaToolAdapter()
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_temperature",
                                "arguments": {"city": "New York"},
                            },
                        }
                    ],
                }
            }
        ]
    }

    calls = adapter.extract_tool_calls(response)
    assert len(calls) == 1
    assert calls[0].name == "get_temperature"
    assert calls[0].arguments == {"city": "New York"}
    # Ollama omits ids; the adapter must synthesize a stable one.
    assert calls[0].id


def test_ollama_adapter_tool_result_uses_tool_name():
    adapter = OllamaToolAdapter()
    call = ToolCall(id="call_0_0_get_temperature", name="get_temperature", arguments={})
    result = ToolResult(tool_call_id=call.id, result="22C", status=ToolResultStatus.SUCCESS)

    msg = adapter.format_tool_result_message(result, call)
    assert msg.role == Role.TOOL
    assert msg.tool_name == "get_temperature"
    assert msg.content == "22C"
    assert msg.tool_call_id is None


def test_ollama_adapter_tool_choice_is_omitted():
    adapter = OllamaToolAdapter()
    assert adapter.format_tool_choice("auto") is None
    assert adapter.format_tool_choice(None) is None


# --------------------------------------------------------------------------- #
# Ollama provider: payload wiring
# --------------------------------------------------------------------------- #


def test_ollama_complete_with_tools_payload(monkeypatch):
    provider = OllamaProvider(ProviderConfig(name="ollama"))
    captured = {}

    def fake_request(url, payload, headers, timeout=None, **kwargs):
        captured["payload"] = payload
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_temperature", "arguments": {"city": "NYC"}}}
                ],
            },
            "prompt_eval_count": 3,
            "eval_count": 4,
            "done": True,
        }

    monkeypatch.setattr(provider, "_make_http_request", fake_request)

    tools = [{"type": "function", "function": {"name": "get_temperature", "parameters": {}}}]
    response = provider.complete_with_tools(
        messages=[Message(role=Role.USER, content="temp in NYC?")],
        tools=tools,
        model="llama3.1:8b",
    )

    assert captured["payload"]["tools"] == tools
    assert captured["payload"]["stream"] is False
    # tool_calls survive into the parsed response for the adapter to read.
    assert response.choices[0].message.tool_calls[0]["function"]["name"] == "get_temperature"


def test_ollama_format_messages_surfaces_tool_name():
    provider = OllamaProvider(ProviderConfig(name="ollama"))
    messages = [
        Message(role=Role.USER, content="hi"),
        Message(role=Role.TOOL, content="22C", tool_name="get_temperature"),
    ]

    formatted = provider._format_messages_for_ollama(messages)
    tool_msg = formatted[1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_name"] == "get_temperature"
    assert "tool_call_id" not in tool_msg


def test_ollama_format_passthrough_structured_output():
    provider = OllamaProvider(ProviderConfig(name="ollama"))
    schema = {"type": "object", "properties": {"x": {"type": "number"}}}

    payload = provider._build_request_payload(
        [Message(role=Role.USER, content="hi")],
        "llama3.1:8b",
        stream=False,
        format=schema,
        max_tokens=64,
    )
    assert payload["format"] == schema
    assert payload["options"]["num_predict"] == 64
