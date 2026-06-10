"""Tests for httpx error wrapping in the HTTP layer."""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import httpx
import pytest

from justllms.core.base import BaseProvider
from justllms.core.models import Message, ModelInfo, ProviderConfig
from justllms.core.streaming import parse_sse_stream
from justllms.exceptions import ProviderError, RequestTimeoutError, raise_from_httpx_error


class MinimalProvider(BaseProvider):
    requires_api_key = False

    def __init__(self) -> None:
        super().__init__(ProviderConfig(name="test"))

    @property
    def name(self) -> str:
        return "test"

    def get_available_models(self) -> Dict[str, ModelInfo]:
        return {"test-model": ModelInfo(name="test-model", provider="test")}

    def complete(
        self,
        messages: List[Message],
        model: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError


class TestRaiseFromHttpxError:
    def test_timeout_maps_to_request_timeout_error(self):
        with pytest.raises(RequestTimeoutError, match="timed out"):
            raise_from_httpx_error(httpx.ReadTimeout("slow"), provider="openai")

    def test_connection_error_maps_to_provider_error(self):
        with pytest.raises(ProviderError, match="failed"):
            raise_from_httpx_error(httpx.ConnectError("refused"), provider="openai")

    def test_non_httpx_error_is_reraised(self):
        with pytest.raises(RuntimeError, match="boom"):
            raise_from_httpx_error(RuntimeError("boom"), provider="openai")


class TestMakeHttpRequestErrors:
    def test_wraps_httpx_timeout(self):
        provider = MinimalProvider()
        timeout_exc = httpx.ReadTimeout("timed out")

        with patch("httpx.Client") as client_cls:
            client = MagicMock()
            client.__enter__.return_value = client
            client.post.side_effect = timeout_exc
            client_cls.return_value = client

            with pytest.raises(RequestTimeoutError, match="test request timed out"):
                provider._make_http_request("https://example.com", {"x": 1})

    def test_wraps_other_httpx_errors(self):
        provider = MinimalProvider()
        connect_exc = httpx.ConnectError("connection refused")

        with patch("httpx.Client") as client_cls:
            client = MagicMock()
            client.__enter__.return_value = client
            client.post.side_effect = connect_exc
            client_cls.return_value = client

            with pytest.raises(ProviderError, match="test request failed"):
                provider._make_http_request("https://example.com", {"x": 1})


class TestStreamingErrors:
    def test_wraps_httpx_timeout(self):
        timeout_exc = httpx.ReadTimeout("timed out")

        with patch("httpx.Client") as client_cls:
            client = MagicMock()
            client.__enter__.return_value = client
            client.stream.side_effect = timeout_exc
            client_cls.return_value = client

            with pytest.raises(RequestTimeoutError, match="Streaming request timed out"):
                list(
                    parse_sse_stream(
                        url="https://example.com",
                        payload={"stream": True},
                        headers={},
                        parse_chunk_fn=lambda _line: None,
                        error_prefix="Streaming request",
                    )
                )
