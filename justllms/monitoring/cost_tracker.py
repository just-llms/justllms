"""Cost tracking functionality."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from justllms.core.models import Usage


class CostTracker:
    """Track costs across providers and models."""

    def __init__(
        self,
        persist_path: Optional[Path] = None,
        budget_daily: Optional[float] = None,
        budget_monthly: Optional[float] = None,
        budget_per_request: Optional[float] = None,
    ):
        self.persist_path = persist_path
        self.budget_daily = budget_daily
        self.budget_monthly = budget_monthly
        self.budget_per_request = budget_per_request

        self.costs: Dict[str, List[Dict[str, Any]]] = {}
        self._load_costs()

    def _load_costs(self) -> None:
        """Load costs from persistence file."""
        if self.persist_path and self.persist_path.exists():
            try:
                with open(self.persist_path) as f:
                    self.costs = json.load(f)
            except Exception:
                self.costs = {}

    def _save_costs(self) -> None:
        """Save costs to persistence file."""
        if self.persist_path:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w") as f:
                json.dump(self.costs, f, indent=2, default=str)

    def track_usage(
        self,
        provider: str,
        model: str,
        usage: Usage,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Track usage and cost for a request."""
        timestamp = datetime.now()

        cost_entry = {
            "timestamp": timestamp.isoformat(),
            "provider": provider,
            "model": model,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "estimated_cost": usage.estimated_cost or 0.0,
            "request_id": request_id,
            "metadata": metadata or {},
        }

        # Add to in-memory tracking
        key = f"{provider}:{model}"
        if key not in self.costs:
            self.costs[key] = []
        self.costs[key].append(cost_entry)

        # Check budgets
        warnings = self._check_budgets(cost_entry)

        # Persist
        self._save_costs()

        return {
            "cost_entry": cost_entry,
            "warnings": warnings,
        }

    def _check_budgets(self, cost_entry: Dict[str, Any]) -> List[str]:
        """Check if budgets are exceeded."""
        warnings = []

        # Per-request budget
        if self.budget_per_request and cost_entry["estimated_cost"] > self.budget_per_request:
            warnings.append(
                f"Request cost ${cost_entry['estimated_cost']:.4f} exceeds "
                f"per-request budget ${self.budget_per_request:.4f}"
            )

        # Daily budget
        if self.budget_daily:
            daily_cost = self.get_cost_summary(period="daily")["total_cost"]
            if daily_cost > self.budget_daily:
                warnings.append(
                    f"Daily cost ${daily_cost:.4f} exceeds daily budget ${self.budget_daily:.4f}"
                )

        # Monthly budget
        if self.budget_monthly:
            monthly_cost = self.get_cost_summary(period="monthly")["total_cost"]
            if monthly_cost > self.budget_monthly:
                warnings.append(
                    f"Monthly cost ${monthly_cost:.4f} exceeds "
                    f"monthly budget ${self.budget_monthly:.4f}"
                )

        return warnings

    def get_cost_summary(  # noqa: C901
        self,
        period: str = "all",  # "all", "daily", "weekly", "monthly"
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get cost summary for a period."""
        now = datetime.now()

        # Determine time filter
        if period == "daily":
            start_time = now - timedelta(days=1)
        elif period == "weekly":
            start_time = now - timedelta(days=7)
        elif period == "monthly":
            start_time = now - timedelta(days=30)
        else:
            start_time = None

        # Filter and aggregate costs
        total_cost = 0.0
        total_tokens = 0
        request_count = 0
        provider_costs = {}
        model_costs = {}

        for key, entries in self.costs.items():
            key_provider, key_model = key.split(":", 1)

            # Apply filters
            if provider and key_provider != provider:
                continue
            if model and key_model != model:
                continue

            for entry in entries:
                # Time filter
                if start_time:
                    entry_time = datetime.fromisoformat(entry["timestamp"])
                    if entry_time < start_time:
                        continue

                cost = entry["estimated_cost"]
                tokens = entry["total_tokens"]

                total_cost += cost
                total_tokens += tokens
                request_count += 1

                # Aggregate by provider
                if key_provider not in provider_costs:
                    provider_costs[key_provider] = {"cost": 0.0, "tokens": 0, "requests": 0}
                provider_costs[key_provider]["cost"] += cost
                provider_costs[key_provider]["tokens"] += tokens
                provider_costs[key_provider]["requests"] += 1

                # Aggregate by model
                if key_model not in model_costs:
                    model_costs[key_model] = {"cost": 0.0, "tokens": 0, "requests": 0}
                model_costs[key_model]["cost"] += cost
                model_costs[key_model]["tokens"] += tokens
                model_costs[key_model]["requests"] += 1

        return {
            "period": period,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "request_count": request_count,
            "average_cost_per_request": total_cost / request_count if request_count > 0 else 0.0,
            "provider_breakdown": provider_costs,
            "model_breakdown": model_costs,
            "filters": {
                "provider": provider,
                "model": model,
            },
        }

    def get_cost_history(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get cost history entries."""
        history = []

        for key, entries in self.costs.items():
            key_provider, key_model = key.split(":", 1)

            # Apply filters
            if provider and key_provider != provider:
                continue
            if model and key_model != model:
                continue

            history.extend(entries)

        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x["timestamp"], reverse=True)

        return history[:limit]

    def clear_old_entries(self, days: int = 30) -> int:
        """Clear entries older than specified days."""
        cutoff_time = datetime.now() - timedelta(days=days)
        removed_count = 0

        for key in list(self.costs.keys()):
            filtered_entries = []

            for entry in self.costs[key]:
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if entry_time >= cutoff_time:
                    filtered_entries.append(entry)
                else:
                    removed_count += 1

            if filtered_entries:
                self.costs[key] = filtered_entries
            else:
                del self.costs[key]

        self._save_costs()
        return removed_count
