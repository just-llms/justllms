"""Tests for OpenAI/Azure reasoning-model parameter adaptation."""

import pytest

from justllms.core.models import Message, ProviderConfig, Role
from justllms.core.openai_base import (
    adapt_payload_for_reasoning_model,
    is_reasoning_model,
)
from justllms.providers.azure_openai import AzureOpenAIProvider
from justllms.providers.openai import OpenAIProvider


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-5", True),
        ("gpt-5-mini", True),
        ("gpt-5-nano", True),
        ("gpt-5.4", True),
        ("gpt-5.4-pro", True),
        ("gpt-5.4-nano", True),
        ("gpt-5.5", True),
        ("o1", True),
        ("o3", True),
        ("o4-mini", True),
        ("openai/gpt-5", True),
        # Non-reasoning chat / legacy models
        ("gpt-5-chat", False),
        ("gpt-4o", False),
        ("gpt-4o-mini", False),
        ("gpt-4.1", False),
        ("gpt-oss-120b", False),
        ("gpt-35-turbo", False),
        ("claude-opus-4-7", False),
    ],
)
def test_is_reasoning_model(model, expected):
    assert is_reasoning_model(model) is expected


def test_adapt_renames_max_tokens_and_drops_sampling():
    payload = {
        "model": "gpt-5",
        "messages": [],
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "logprobs": True,
        "top_logprobs": 5,
        "logit_bias": {"1": 1},
        "reasoning_effort": "high",
    }

    adapted = adapt_payload_for_reasoning_model(payload, "gpt-5")

    assert adapted["max_completion_tokens"] == 500
    assert "max_tokens" not in adapted
    for dropped in (
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "logprobs",
        "top_logprobs",
        "logit_bias",
    ):
        assert dropped not in adapted
    # Reasoning-specific params survive.
    assert adapted["reasoning_effort"] == "high"


def test_adapt_noop_for_non_reasoning_model():
    payload = {
        "model": "gpt-4o",
        "max_tokens": 500,
        "temperature": 0.7,
    }
    adapted = adapt_payload_for_reasoning_model(payload, "gpt-4o")
    # Returned unchanged (same object) for non-reasoning models.
    assert adapted is payload
    assert adapted["max_tokens"] == 500
    assert adapted["temperature"] == 0.7


def _openai_provider():
    return OpenAIProvider(ProviderConfig(name="openai", api_key="test-key"))


def test_openai_build_payload_reasoning_model():
    provider = _openai_provider()
    messages = [Message(role=Role.USER, content="hi")]

    payload = provider._build_base_payload(
        messages, "gpt-5", max_tokens=256, temperature=0.5, reasoning_effort="low"
    )

    assert payload["max_completion_tokens"] == 256
    assert "max_tokens" not in payload
    assert "temperature" not in payload
    assert payload["reasoning_effort"] == "low"


def test_openai_build_payload_regular_model():
    provider = _openai_provider()
    messages = [Message(role=Role.USER, content="hi")]

    payload = provider._build_base_payload(messages, "gpt-4o", max_tokens=256, temperature=0.5)

    assert payload["max_tokens"] == 256
    assert payload["temperature"] == 0.5
    assert "max_completion_tokens" not in payload


def test_azure_complete_adapts_payload(monkeypatch):
    provider = AzureOpenAIProvider(
        ProviderConfig(name="azure_openai", api_key="test-key", resource_name="my-res")
    )

    captured = {}

    def fake_request(url, payload, headers, timeout=None, **kwargs):
        captured["payload"] = payload
        return {
            "id": "x",
            "model": "gpt-5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(provider, "_make_http_request", fake_request)

    provider.complete(
        [Message(role=Role.USER, content="hi")],
        "gpt-5",
        max_tokens=128,
        temperature=0.9,
    )

    payload = captured["payload"]
    assert payload["max_completion_tokens"] == 128
    assert "max_tokens" not in payload
    assert "temperature" not in payload


def test_azure_parse_response_accepts_null_content():
    """Azure tool-call responses set content to null; parsing must not crash."""
    provider = AzureOpenAIProvider(
        ProviderConfig(name="azure_openai", api_key="test-key", resource_name="my-res")
    )

    response = provider._parse_response(
        {
            "id": "chatcmpl-x",
            "model": "gpt-4o",
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
    )

    message = response.choices[0].message
    assert message.content == ""
    assert message.tool_calls[0]["function"]["name"] == "get_weather"
