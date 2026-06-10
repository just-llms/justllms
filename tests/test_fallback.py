"""Tests for failure-driven provider fallback."""

from typing import Any, Callable, Dict, List, Optional

import httpx
import pytest

from justllms.core.base import BaseProvider, BaseResponse
from justllms.core.client import Client
from justllms.core.models import Choice, Message, ModelInfo, ProviderConfig, Role, Usage
from justllms.exceptions import (
    AuthenticationError,
    ConfigurationError,
    ProviderError,
    RateLimitError,
    ValidationError,
)
from justllms.exceptions import RequestTimeoutError
from justllms.reliability import FallbackAttempt, FallbackExecutor, classify_error


def make_response(model: str, content: str = "ok") -> BaseResponse:
    """Build a minimal canned provider response."""
    return BaseResponse(
        id="resp-1",
        model=model,
        choices=[
            Choice(
                index=0,
                message=Message(role=Role.ASSISTANT, content=content),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


class FakeProvider(BaseProvider):
    """Configurable in-memory provider for tests."""

    requires_api_key = False

    def __init__(
        self,
        name: str,
        models: List[str],
        behavior: Optional[Callable[[str], BaseResponse]] = None,
    ):
        super().__init__(ProviderConfig(name=name))
        self._name = name
        self._models = models
        self._behavior = behavior or (lambda model: make_response(model))
        self.calls: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    def get_available_models(self) -> Dict[str, ModelInfo]:
        return {m: ModelInfo(name=m, provider=self._name) for m in self._models}

    def complete(
        self,
        messages: List[Message],
        model: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> BaseResponse:
        self.calls.append({"model": model, "kwargs": kwargs})
        return self._behavior(model)


def always_raise(exc: Exception) -> Callable[[str], BaseResponse]:
    def _behavior(model: str) -> BaseResponse:
        raise exc

    return _behavior


# ---------------------------------------------------------------------------
# classify_error
# ---------------------------------------------------------------------------


class TestClassifyError:
    def test_rate_limit_error_instance(self):
        assert classify_error(RateLimitError("slow down")) == "rate_limit"

    def test_provider_error_429(self):
        assert classify_error(ProviderError("err", status_code=429)) == "rate_limit"

    @pytest.mark.parametrize("status", [500, 502, 503, 599])
    def test_provider_error_5xx(self, status):
        assert classify_error(ProviderError("err", status_code=status)) == "server_error"

    def test_timeout_error_instance(self):
        assert classify_error(RequestTimeoutError("too slow")) == "timeout"

    def test_httpx_timeout(self):
        assert classify_error(httpx.ConnectTimeout("timed out")) == "timeout"

    def test_provider_error_408(self):
        assert classify_error(ProviderError("err", status_code=408)) == "timeout"

    def test_httpx_connect_error(self):
        assert classify_error(httpx.ConnectError("refused")) == "connection"

    def test_httpx_request_error(self):
        assert classify_error(httpx.RequestError("network")) == "connection"

    def test_authentication_error_instance(self):
        assert classify_error(AuthenticationError("bad key")) == "auth"

    @pytest.mark.parametrize("status", [401, 403])
    def test_provider_error_auth_status(self, status):
        assert classify_error(ProviderError("err", status_code=status)) == "auth"

    @pytest.mark.parametrize("status", [400, 404, 422])
    def test_provider_error_other_4xx_not_failover(self, status):
        assert classify_error(ProviderError("err", status_code=status)) is None

    def test_provider_error_without_status(self):
        assert classify_error(ProviderError("err")) is None

    def test_validation_error(self):
        assert classify_error(ValidationError("bad input")) is None

    def test_configuration_error(self):
        assert classify_error(ConfigurationError("bad config")) is None

    def test_plain_value_error(self):
        assert classify_error(ValueError("nope")) is None


# ---------------------------------------------------------------------------
# build_chain
# ---------------------------------------------------------------------------


class TestBuildChain:
    def make_providers(self) -> Dict[str, BaseProvider]:
        return {
            "a": FakeProvider("a", ["model-a1", "model-a2"]),
            "b": FakeProvider("b", ["model-b"]),
            "c": FakeProvider("c", ["model-c"]),
        }

    def test_explicit_chain_with_model_and_bare_provider(self):
        executor = FallbackExecutor({"fallback_chain": ["b/model-b", "c"]})
        chain = executor.build_chain(("a", "model-a1"), self.make_providers())
        assert chain == [("a", "model-a1"), ("b", "model-b"), ("c", "model-c")]

    def test_skips_unknown_provider_and_invalid_model(self):
        executor = FallbackExecutor(
            {"fallback_chain": ["nope/model-x", "b/wrong-model", "c/model-c"]}
        )
        chain = executor.build_chain(("a", "model-a1"), self.make_providers())
        assert chain == [("a", "model-a1"), ("c", "model-c")]

    def test_dedupes_primary(self):
        executor = FallbackExecutor({"fallback_chain": ["a/model-a1", "b/model-b"]})
        chain = executor.build_chain(("a", "model-a1"), self.make_providers())
        assert chain == [("a", "model-a1"), ("b", "model-b")]

    def test_auto_fallback_derives_chain_from_other_providers(self):
        executor = FallbackExecutor({"auto_fallback": True})
        chain = executor.build_chain(("b", "model-b"), self.make_providers())
        assert chain[0] == ("b", "model-b")
        assert set(chain[1:]) == {("a", "model-a1"), ("c", "model-c")}

    def test_auto_fallback_ignored_when_chain_present(self):
        executor = FallbackExecutor({"auto_fallback": True, "fallback_chain": ["c"]})
        chain = executor.build_chain(("a", "model-a1"), self.make_providers())
        assert chain == [("a", "model-a1"), ("c", "model-c")]

    def test_max_fallback_attempts_truncation(self):
        executor = FallbackExecutor(
            {"fallback_chain": ["b/model-b", "c/model-c"], "max_fallback_attempts": 2}
        )
        chain = executor.build_chain(("a", "model-a1"), self.make_providers())
        assert chain == [("a", "model-a1"), ("b", "model-b")]

    def test_enabled_property(self):
        assert not FallbackExecutor({}).enabled
        assert FallbackExecutor({"fallback_chain": ["b"]}).enabled
        assert FallbackExecutor({"auto_fallback": True}).enabled


# ---------------------------------------------------------------------------
# FallbackExecutor.execute
# ---------------------------------------------------------------------------


class TestExecute:
    def test_success_first_try(self):
        executor = FallbackExecutor({"fallback_chain": ["b/model-b"]})
        chain = [("a", "model-a"), ("b", "model-b")]

        def call(provider: str, model: str) -> Any:
            return f"{provider}:{model}"

        result, attempts = executor.execute(chain, call)
        assert result == "a:model-a"
        assert len(attempts) == 1
        assert attempts[0].succeeded is True
        assert attempts[0].error is None
        assert not any(not a.succeeded for a in attempts)

    def test_first_fails_429_second_succeeds(self):
        executor = FallbackExecutor({"fallback_chain": ["b/model-b"]})
        chain = [("a", "model-a"), ("b", "model-b")]

        def call(provider: str, model: str) -> Any:
            if provider == "a":
                raise ProviderError("rate limited", status_code=429)
            return f"{provider}:{model}"

        result, attempts = executor.execute(chain, call)
        assert result == "b:model-b"
        assert len(attempts) == 2
        assert attempts[0].provider == "a"
        assert attempts[0].succeeded is False
        assert attempts[0].error_category == "rate_limit"
        assert "rate limited" in attempts[0].error
        assert attempts[1].provider == "b"
        assert attempts[1].succeeded is True

    def test_all_fail_raises_last_error(self):
        executor = FallbackExecutor({"fallback_chain": ["b/model-b"]})
        chain = [("a", "model-a"), ("b", "model-b")]

        def call(provider: str, model: str) -> Any:
            if provider == "a":
                raise ProviderError("a down", status_code=503)
            raise ProviderError("b down", status_code=500)

        with pytest.raises(ProviderError, match="b down"):
            executor.execute(chain, call)

    def test_non_retryable_error_raised_immediately(self):
        executor = FallbackExecutor({"fallback_chain": ["b/model-b"]})
        chain = [("a", "model-a"), ("b", "model-b")]
        calls: List[str] = []

        def call(provider: str, model: str) -> Any:
            calls.append(provider)
            raise ProviderError("bad request", status_code=400)

        with pytest.raises(ProviderError, match="bad request"):
            executor.execute(chain, call)
        assert calls == ["a"]

    def test_category_filtering_via_fallback_on(self):
        executor = FallbackExecutor(
            {"fallback_chain": ["b/model-b"], "fallback_on": ["server_error"]}
        )
        chain = [("a", "model-a"), ("b", "model-b")]
        calls: List[str] = []

        def call(provider: str, model: str) -> Any:
            calls.append(provider)
            raise ProviderError("rate limited", status_code=429)

        with pytest.raises(ProviderError, match="rate limited"):
            executor.execute(chain, call)
        assert calls == ["a"]

    def test_empty_chain_raises(self):
        executor = FallbackExecutor({})
        with pytest.raises(ValueError, match="empty"):
            executor.execute([], lambda p, m: None)

    def test_attempt_to_dict(self):
        attempt = FallbackAttempt(
            provider="a",
            model="m",
            error="boom",
            error_category="server_error",
            succeeded=False,
            duration_ms=1.5,
        )
        assert attempt.to_dict() == {
            "provider": "a",
            "model": "m",
            "error": "boom",
            "error_category": "server_error",
            "succeeded": False,
            "duration_ms": 1.5,
        }


# ---------------------------------------------------------------------------
# Client end-to-end
# ---------------------------------------------------------------------------


class TestClientFallback:
    def make_client(
        self,
        routing: Optional[Dict[str, Any]] = None,
        provider_a: Optional[FakeProvider] = None,
        provider_b: Optional[FakeProvider] = None,
    ) -> Client:
        provider_a = provider_a or FakeProvider(
            "a",
            ["model-a"],
            behavior=always_raise(ProviderError("rate limited", status_code=429)),
        )
        provider_b = provider_b or FakeProvider("b", ["model-b"])
        config = {
            "providers": {"a": {"enabled": True}, "b": {"enabled": True}},
            "routing": routing if routing is not None else {"fallback_chain": ["b/model-b"]},
        }
        return Client(config, providers={"a": provider_a, "b": provider_b})

    def test_failover_to_second_provider(self):
        client = self.make_client()
        response = client.completion.create(
            messages=[{"role": "user", "content": "hi"}],
            model="a/model-a",
        )
        assert response.provider == "b"
        assert response.model == "model-b"
        assert response.fallback_used is True
        assert response.fallback_attempts is not None
        assert len(response.fallback_attempts) == 2
        assert response.fallback_attempts[0]["provider"] == "a"
        assert response.fallback_attempts[0]["succeeded"] is False
        assert response.fallback_attempts[0]["error_category"] == "rate_limit"
        assert response.fallback_attempts[1]["provider"] == "b"
        assert response.fallback_attempts[1]["succeeded"] is True

    def test_explicit_provider_failover(self):
        client = self.make_client()
        response = client.completion.create(
            messages=[{"role": "user", "content": "hi"}],
            provider="a",
        )
        assert response.provider == "b"
        assert response.fallback_used is True

    def test_no_failover_metadata_when_primary_succeeds(self):
        provider_a = FakeProvider("a", ["model-a"])
        client = self.make_client(provider_a=provider_a)
        response = client.completion.create(
            messages=[{"role": "user", "content": "hi"}],
            model="a/model-a",
        )
        assert response.provider == "a"
        assert response.fallback_used is False
        assert response.fallback_attempts is None

    def test_per_request_fallback_false_disables_failover(self):
        client = self.make_client()
        with pytest.raises(ProviderError, match="rate limited"):
            client.completion.create(
                messages=[{"role": "user", "content": "hi"}],
                model="a/model-a",
                fallback=False,
            )

    def test_non_retryable_error_no_failover(self):
        provider_a = FakeProvider(
            "a",
            ["model-a"],
            behavior=always_raise(ProviderError("bad request", status_code=400)),
        )
        provider_b = FakeProvider("b", ["model-b"])
        client = self.make_client(provider_a=provider_a, provider_b=provider_b)
        with pytest.raises(ProviderError, match="bad request"):
            client.completion.create(
                messages=[{"role": "user", "content": "hi"}],
                model="a/model-a",
            )
        assert provider_b.calls == []

    def test_no_fallback_config_keeps_current_behavior(self):
        client = self.make_client(routing={})
        with pytest.raises(ProviderError, match="rate limited"):
            client.completion.create(
                messages=[{"role": "user", "content": "hi"}],
                model="a/model-a",
            )

    def test_fallback_kwarg_not_forwarded_to_provider(self):
        provider_a = FakeProvider("a", ["model-a"])
        client = self.make_client(provider_a=provider_a)
        client.completion.create(
            messages=[{"role": "user", "content": "hi"}],
            model="a/model-a",
            fallback=True,
        )
        assert provider_a.calls
        assert "fallback" not in provider_a.calls[0]["kwargs"]

    def test_auto_fallback_derivation(self):
        provider_a = FakeProvider(
            "a",
            ["model-a"],
            behavior=always_raise(ProviderError("down", status_code=503)),
        )
        client = self.make_client(routing={"auto_fallback": True}, provider_a=provider_a)
        response = client.completion.create(
            messages=[{"role": "user", "content": "hi"}],
            model="a/model-a",
        )
        assert response.provider == "b"
        assert response.fallback_used is True
