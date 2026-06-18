"""Budget limits enforced against accumulated usage.

Budget checks are *pre-flight*: :meth:`BudgetManager.check` is called before
each request and compares the limits against usage accumulated so far. This
means the request that crosses a limit completes normally, and the *next*
request is blocked (or warned about). This keeps enforcement cheap and avoids
having to predict a request's cost up front.
"""

import logging
from typing import Optional, Set

from pydantic import BaseModel, ConfigDict

from justllms.exceptions import BudgetExceededError
from justllms.monitoring.usage import UsageTracker

logger = logging.getLogger(__name__)


class BudgetConfig(BaseModel):
    """Configuration for budget limits.

    All limits are cumulative for the lifetime of the client's UsageTracker
    (i.e. since client creation or the last ``reset_usage()`` call).
    """

    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    max_cost: Optional[float] = None
    """Maximum cumulative estimated cost in USD."""

    max_tokens: Optional[int] = None
    """Maximum cumulative total tokens."""

    max_requests: Optional[int] = None
    """Maximum cumulative request count."""

    on_exceeded: str = "raise"
    """What to do when a limit is reached: "raise" (default) or "warn"."""


class BudgetManager:
    """Enforces budget limits against a UsageTracker.

    Semantics: ``check()`` is a pre-flight gate. It raises (or warns) when a
    limit has *already* been reached by previous requests, so the request that
    crosses a limit is allowed to complete and subsequent requests are blocked.
    """

    def __init__(self, config: BudgetConfig, tracker: UsageTracker):
        """Initialize the budget manager.

        Args:
            config: Budget limits and behavior configuration.
            tracker: UsageTracker holding the accumulated usage to check against.
        """
        self.config = config
        self.tracker = tracker
        self._warned_types: Set[str] = set()

    def check(self) -> None:
        """Check accumulated usage against configured limits.

        Call before dispatching a request. No-op if budgets are disabled or
        no limits are configured.

        Raises:
            BudgetExceededError: If a limit is reached and on_exceeded="raise".
        """
        if not self.config.enabled:
            return

        if self.config.max_cost is not None:
            current_cost = self.tracker.total_cost
            if current_cost >= self.config.max_cost:
                self._handle_breach("cost", self.config.max_cost, current_cost)

        if self.config.max_tokens is not None:
            current_tokens = self.tracker.total_tokens
            if current_tokens >= self.config.max_tokens:
                self._handle_breach("tokens", self.config.max_tokens, current_tokens)

        if self.config.max_requests is not None:
            current_requests = self.tracker.request_count
            if current_requests >= self.config.max_requests:
                self._handle_breach("requests", self.config.max_requests, current_requests)

    def _handle_breach(self, limit_type: str, limit: float, current: float) -> None:
        """Raise or warn (once per breach type) about an exceeded limit."""
        if self.config.on_exceeded == "warn":
            if limit_type not in self._warned_types:
                self._warned_types.add(limit_type)
                logger.warning(
                    "Budget limit reached for %s: current=%s, limit=%s. "
                    "Requests will continue because on_exceeded='warn'.",
                    limit_type,
                    current,
                    limit,
                )
            return

        raise BudgetExceededError(
            f"Budget limit reached for {limit_type}: current={current}, limit={limit}",
            limit_type=limit_type,
            limit=limit,
            current=current,
        )

    def reset(self) -> None:
        """Clear warn-once state after usage is reset."""
        self._warned_types.clear()
