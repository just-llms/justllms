"""Tests for provider alias registration on the client."""

import os
from unittest.mock import patch

import pytest

from justllms.core.client import Client
from justllms.core.models import ProviderConfig
from justllms.providers.grok import GrokProvider


def test_xai_alias_registered_when_grok_initialized_from_env() -> None:
    """XAI_API_KEY configures grok; xai alias must resolve for SXS and explicit calls."""
    with patch.dict(os.environ, {"XAI_API_KEY": "test-key"}, clear=False):
        client = Client()

    assert "grok" in client.providers
    assert "xai" in client.providers
    assert client.providers["xai"] is client.providers["grok"]


def test_grok_alias_registered_when_xai_configured() -> None:
    """YAML/config using the xai key should also expose grok as an alias."""
    client = Client(config={"providers": {"xai": {"api_key": "test-key"}}})

    assert "xai" in client.providers
    assert "grok" in client.providers
    assert client.providers["grok"] is client.providers["xai"]


def test_add_provider_registers_aliases() -> None:
    """Manually added providers should expose registered alternate names."""
    client = Client(config={"providers": {}})
    grok = GrokProvider(ProviderConfig(name="grok", api_key="test-key"))
    client.add_provider("grok", grok)

    assert "xai" in client.providers
    assert client.get_provider("xai") is grok


def test_existing_alias_not_overwritten() -> None:
    """A separately configured alias must not be replaced by auto-registration."""
    client = Client(
        config={
            "providers": {
                "grok": {"api_key": "grok-key"},
                "xai": {"api_key": "xai-key"},
            }
        }
    )

    assert client.providers["grok"] is not client.providers["xai"]
    assert client.providers["grok"].config.api_key == "grok-key"
    assert client.providers["xai"].config.api_key == "xai-key"
