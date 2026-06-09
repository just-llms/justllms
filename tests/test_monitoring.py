"""Tests for usage tracking and budget limits."""

import logging
import threading
from typing import Any, Dict, Iterator, List, Optional

import pytest

from justllms import Client
from justllms.core.base import BaseProvider, BaseResponse
from justllms.core.models import Choice, Message, ModelInfo, ProviderConfig, Role, Usage
from justllms.core.streaming import StreamChunk, SyncStreamResponse
from justllms.exceptions import BudgetExceededError
from justllms.monitoring import BudgetConfig, BudgetManager, UsageTracker

# Cost per call with the fake provider's pricing and usage below:
# 100/1000 * 0.01 + 50/1000 * 0.02 = 0.002
EXPECTED_COST_PER_CALL = 0.002


def make_usage(prompt: int = 100, completion: int = 50, cost: Optional[float] = None) -> Usage:
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        estimated_cost=cost,
    )


class FakeProvider(BaseProvider):
    """Minimal provider returning canned responses with usage."""

    requires_api_key = False

    @property
    def name(self) -> str:
        return "fake"

    def get_available_models(self) -> Dict[str, ModelInfo]:
        return {
            "fake-model": ModelInfo(
                name="fake-model",
                provider="fake",
                cost_per_1k_prompt_tokens=0.01,
                cost_per_1k_completion_tokens=0.02,
            )
        }

    def complete(
        self,
        messages: List[Message],
        model: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> BaseResponse:
        return BaseResponse(
            id="resp-1",
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role=Role.ASSISTANT, content="hello"),
                    finish_reason="stop",
                )
            ],
            usage=make_usage(),
        )


class FakeStreamingProvider(FakeProvider):
    """Fake provider that supports streaming."""

    def supports_streaming(self) -> bool:
        return True

    def supports_streaming_for_model(self, model: str) -> bool:
        return True

    def stream(
        self,
        messages: List[Message],
        model: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> SyncStreamResponse:
        def chunks() -> Iterator[StreamChunk]:
            yield StreamChunk(content="hel")
            yield StreamChunk(content="lo", finish_reason="stop", usage=make_usage())

        return SyncStreamResponse(
            provider=self, model=model, messages=messages, raw_stream=chunks()
        )


def make_client(
    provider: Optional[BaseProvider] = None, config: Optional[Dict[str, Any]] = None
) -> Client:
    provider = provider or FakeProvider(ProviderConfig(name="fake"))
    return Client(config=config or {"providers": {}}, providers={"fake": provider})


MESSAGES = [{"role": "user", "content": "hi"}]


class TestUsageTracker:
    def test_record_and_totals(self) -> None:
        tracker = UsageTracker()
        tracker.record("openai", "gpt-4o", make_usage(100, 50, cost=0.5))
        tracker.record("openai", "gpt-4o-mini", make_usage(10, 5, cost=0.1))
        tracker.record("google", "gemini-pro", make_usage(20, 10, cost=0.2), streamed=True)

        assert tracker.request_count == 3
        assert tracker.total_prompt_tokens == 130
        assert tracker.total_completion_tokens == 65
        assert tracker.total_tokens == 195
        assert tracker.total_cost == pytest.approx(0.8)

        records = tracker.get_records()
        assert len(records) == 3
        assert records[2].streamed is True
        assert records[0].provider == "openai"
        assert records[0].cost == pytest.approx(0.5)

    def test_summary_breakdowns(self) -> None:
        tracker = UsageTracker()
        tracker.record("openai", "gpt-4o", make_usage(100, 50, cost=0.5))
        tracker.record("openai", "gpt-4o", make_usage(100, 50, cost=0.5))
        tracker.record("google", "gemini-pro", make_usage(20, 10, cost=0.2))

        summary = tracker.get_summary()
        assert summary["request_count"] == 3
        assert summary["total_cost"] == pytest.approx(1.2)
        assert summary["by_provider"]["openai"]["requests"] == 2
        assert summary["by_provider"]["openai"]["cost"] == pytest.approx(1.0)
        assert summary["by_provider"]["google"]["total_tokens"] == 30
        assert summary["by_model"]["openai/gpt-4o"]["requests"] == 2
        assert summary["by_model"]["google/gemini-pro"]["cost"] == pytest.approx(0.2)

    def test_none_usage_tolerated(self) -> None:
        tracker = UsageTracker()
        tracker.record("openai", "gpt-4o", None)
        tracker.record("openai", "gpt-4o", make_usage(10, 5))  # no estimated_cost

        assert tracker.request_count == 2
        assert tracker.total_tokens == 15
        assert tracker.total_cost == 0.0

    def test_reset(self) -> None:
        tracker = UsageTracker()
        tracker.record("openai", "gpt-4o", make_usage(100, 50, cost=0.5))
        tracker.reset()

        assert tracker.request_count == 0
        assert tracker.total_cost == 0.0
        assert tracker.get_records() == []
        assert tracker.get_summary()["by_provider"] == {}

    def test_max_records_trimming_keeps_totals(self) -> None:
        tracker = UsageTracker(max_records=5)
        for _ in range(10):
            tracker.record("openai", "gpt-4o", make_usage(10, 5, cost=0.01))

        assert len(tracker.get_records()) == 5
        assert tracker.request_count == 10
        assert tracker.total_tokens == 150
        assert tracker.total_cost == pytest.approx(0.1)

    def test_concurrent_records(self) -> None:
        tracker = UsageTracker()
        n_threads, n_records = 8, 50

        def worker() -> None:
            for _ in range(n_records):
                tracker.record("openai", "gpt-4o", make_usage(10, 5, cost=0.01))

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = n_threads * n_records
        assert tracker.request_count == total
        assert tracker.total_tokens == total * 15
        assert tracker.total_cost == pytest.approx(total * 0.01)
        assert tracker.get_summary()["by_provider"]["openai"]["requests"] == total


class TestBudgetManager:
    def test_no_limits_never_raises(self) -> None:
        tracker = UsageTracker()
        manager = BudgetManager(BudgetConfig(), tracker)
        for _ in range(5):
            tracker.record("openai", "gpt-4o", make_usage(1000, 1000, cost=100.0))
            manager.check()

    def test_max_cost_exceeded(self) -> None:
        tracker = UsageTracker()
        manager = BudgetManager(BudgetConfig(max_cost=1.0), tracker)
        manager.check()  # under limit, fine

        tracker.record("openai", "gpt-4o", make_usage(100, 50, cost=1.5))
        with pytest.raises(BudgetExceededError) as exc_info:
            manager.check()
        assert exc_info.value.limit_type == "cost"
        assert exc_info.value.limit == 1.0
        assert exc_info.value.current == pytest.approx(1.5)

    def test_max_tokens_exceeded(self) -> None:
        tracker = UsageTracker()
        manager = BudgetManager(BudgetConfig(max_tokens=100), tracker)
        tracker.record("openai", "gpt-4o", make_usage(80, 40))
        with pytest.raises(BudgetExceededError) as exc_info:
            manager.check()
        assert exc_info.value.limit_type == "tokens"
        assert exc_info.value.current == 120

    def test_max_requests_exceeded(self) -> None:
        tracker = UsageTracker()
        manager = BudgetManager(BudgetConfig(max_requests=2), tracker)
        tracker.record("openai", "gpt-4o", make_usage())
        manager.check()
        tracker.record("openai", "gpt-4o", make_usage())
        with pytest.raises(BudgetExceededError) as exc_info:
            manager.check()
        assert exc_info.value.limit_type == "requests"

    def test_disabled_budget_never_raises(self) -> None:
        tracker = UsageTracker()
        manager = BudgetManager(BudgetConfig(enabled=False, max_requests=1), tracker)
        tracker.record("openai", "gpt-4o", make_usage())
        manager.check()

    def test_warn_mode_logs_once_per_type(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = UsageTracker()
        manager = BudgetManager(BudgetConfig(max_requests=1, on_exceeded="warn"), tracker)
        tracker.record("openai", "gpt-4o", make_usage())

        with caplog.at_level(logging.WARNING, logger="justllms.monitoring.budget"):
            manager.check()
            manager.check()

        warnings = [r for r in caplog.records if "Budget limit reached" in r.message]
        assert len(warnings) == 1
        assert "requests" in warnings[0].message


class TestClientIntegration:
    def test_usage_recorded_for_completions(self) -> None:
        client = make_client()
        for _ in range(2):
            response = client.completion.create(
                messages=MESSAGES, provider="fake", model="fake-model"
            )
            assert response.usage is not None
            assert response.usage.estimated_cost == pytest.approx(EXPECTED_COST_PER_CALL)

        assert client.usage.request_count == 2
        assert client.usage.total_cost == pytest.approx(2 * EXPECTED_COST_PER_CALL)

        summary = client.get_usage_summary()
        assert summary["request_count"] == 2
        assert summary["by_model"]["fake/fake-model"]["requests"] == 2

        client.reset_usage()
        assert client.usage.request_count == 0

    def test_budget_blocks_second_request(self) -> None:
        client = make_client(config={"providers": {}, "budget": {"max_requests": 1}})
        client.completion.create(messages=MESSAGES, provider="fake", model="fake-model")

        with pytest.raises(BudgetExceededError) as exc_info:
            client.completion.create(messages=MESSAGES, provider="fake", model="fake-model")
        assert exc_info.value.limit_type == "requests"
        assert client.usage.request_count == 1

    def test_budget_warn_mode_allows_requests(self, caplog: pytest.LogCaptureFixture) -> None:
        client = make_client(
            config={"providers": {}, "budget": {"max_requests": 1, "on_exceeded": "warn"}}
        )
        with caplog.at_level(logging.WARNING, logger="justllms.monitoring.budget"):
            client.completion.create(messages=MESSAGES, provider="fake", model="fake-model")
            client.completion.create(messages=MESSAGES, provider="fake", model="fake-model")

        assert client.usage.request_count == 2
        assert any("Budget limit reached" in r.message for r in caplog.records)

    def test_budget_max_cost_via_client(self) -> None:
        client = make_client(
            config={"providers": {}, "budget": {"max_cost": EXPECTED_COST_PER_CALL / 2}}
        )
        client.completion.create(messages=MESSAGES, provider="fake", model="fake-model")
        with pytest.raises(BudgetExceededError) as exc_info:
            client.completion.create(messages=MESSAGES, provider="fake", model="fake-model")
        assert exc_info.value.limit_type == "cost"

    def test_streaming_usage_recorded(self) -> None:
        client = make_client(provider=FakeStreamingProvider(ProviderConfig(name="fake")))
        stream = client.completion.create(
            messages=MESSAGES, provider="fake", model="fake-model", stream=True
        )
        content = "".join(chunk.content or "" for chunk in stream)
        assert content == "hello"

        final = stream.get_final_response()
        assert final.usage is not None

        # Calling get_final_response again must not double-record.
        stream.get_final_response()

        assert client.usage.request_count == 1
        records = client.usage.get_records()
        assert records[0].streamed is True
        assert records[0].total_tokens == 150
        assert client.usage.total_cost == pytest.approx(EXPECTED_COST_PER_CALL)


class TestSyncStreamOnComplete:
    def _make_stream(self, on_complete: Any) -> SyncStreamResponse:
        provider = FakeStreamingProvider(ProviderConfig(name="fake"))

        def chunks() -> Iterator[StreamChunk]:
            yield StreamChunk(content="hi", finish_reason="stop", usage=make_usage())

        return SyncStreamResponse(
            provider=provider,
            model="fake-model",
            messages=[Message(role=Role.USER, content="hi")],
            raw_stream=chunks(),
            on_complete=on_complete,
        )

    def test_on_complete_fires_exactly_once(self) -> None:
        calls: List[Any] = []
        stream = self._make_stream(calls.append)

        final = stream.get_final_response()
        stream.get_final_response()
        stream.get_final_response()

        assert len(calls) == 1
        assert calls[0].usage.total_tokens == final.usage.total_tokens

    def test_on_complete_errors_are_swallowed(self) -> None:
        def boom(_response: Any) -> None:
            raise RuntimeError("callback exploded")

        stream = self._make_stream(boom)
        final = stream.get_final_response()  # must not raise
        assert final.content == "hi"
