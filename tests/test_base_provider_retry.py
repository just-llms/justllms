"""Tests that _make_http_request honors ProviderConfig.max_retries/retry_delay."""

import httpx
import pytest

from justllms.core.models import ProviderConfig
from justllms.providers.anthropic import AnthropicProvider


class _FakeClient:
    """Stands in for httpx.Client and always raises a retryable error."""

    def __init__(self, calls, *args, **kwargs):
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def post(self, *args, **kwargs):
        self._calls.append(1)
        raise httpx.ConnectError("connection refused")


def test_make_http_request_uses_configured_max_retries(monkeypatch):
    provider = AnthropicProvider(
        ProviderConfig(name="anthropic", api_key="test", max_retries=2, retry_delay=0.01)
    )

    calls = []
    monkeypatch.setattr(
        "justllms.core.base.httpx.Client",
        lambda *a, **kw: _FakeClient(calls, *a, **kw),
    )

    with pytest.raises(httpx.ConnectError):
        provider._make_http_request("http://example.com", {"foo": "bar"})

    assert len(calls) == provider.config.max_retries == 2
