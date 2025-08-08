"""Configuration management for JustLLMs."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml  # type: ignore
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class ProviderConfig(BaseModel):
    """Configuration for a single provider."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    organization: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0


class RoutingConfig(BaseModel):
    """Configuration for routing."""

    model_config = ConfigDict(extra="allow")

    strategy: str = "cost"  # "cost", "latency", "quality", "task"
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None
    strategy_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class CacheConfig(BaseModel):
    """Configuration for caching."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    backend: str = "memory"  # "memory", "disk"
    ttl: int = 3600  # 1 hour
    ignore_params: list[str] = Field(default_factory=lambda: ["user", "seed"])
    backend_config: Dict[str, Any] = Field(default_factory=dict)


class MonitoringConfig(BaseModel):
    """Configuration for monitoring."""

    model_config = ConfigDict(extra="allow")

    logging: Dict[str, Any] = Field(
        default_factory=lambda: {
            "level": "INFO",
            "console_output": True,
            "rich_formatting": True,
        }
    )
    cost_tracking: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class ConversationsConfig(BaseModel):
    """Configuration for conversation management."""

    model_config = ConfigDict(extra="allow")

    # Storage backend configuration
    backend: str = "memory"  # "memory", "disk", "redis"
    config: Dict[str, Any] = Field(default_factory=dict)

    # Default conversation settings
    default_model: Optional[str] = None
    default_provider: Optional[str] = None
    max_context_tokens: int = 8000
    context_strategy: str = "truncate"  # "truncate", "summarize", "compress"
    auto_save: bool = True
    auto_title: bool = True
    enable_analytics: bool = True


class Config(BaseModel):
    """Main configuration class for JustLLMs."""

    model_config = ConfigDict(extra="allow")

    providers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    conversations: ConversationsConfig = Field(default_factory=ConversationsConfig)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            if path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif path.suffix == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {path.suffix}")

        return cls(**data)

    @classmethod
    def from_env(cls, prefix: str = "JUSTLLMS_") -> "Config":  # noqa: C901
        """Load configuration from environment variables."""
        load_dotenv()

        config = cls()

        # Load provider API keys from environment
        provider_env_vars = {
            "OPENAI_API_KEY": ("openai", "api_key"),
            "ANTHROPIC_API_KEY": ("anthropic", "api_key"),
            "GOOGLE_API_KEY": ("google", "api_key"),
            "GEMINI_API_KEY": ("google", "api_key"),
            "XAI_API_KEY": ("xai", "api_key"),
            "GROK_API_KEY": ("xai", "api_key"),
            "COHERE_API_KEY": ("cohere", "api_key"),
            "REPLICATE_API_TOKEN": ("replicate", "api_key"),
        }

        for env_var, (provider, key) in provider_env_vars.items():
            value = os.getenv(env_var)
            if value:
                if provider not in config.providers:
                    config.providers[provider] = {}
                config.providers[provider][key] = value

        # Load other environment variables with prefix
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Parse the key structure (e.g., JUSTLLMS_CACHE_ENABLED)
                parts = key[len(prefix) :].lower().split("_")

                if len(parts) >= 2:
                    if parts[0] == "cache" and hasattr(config.cache, parts[1]):
                        # Handle boolean values
                        parsed_value: Any = value
                        if value.lower() in ["true", "false"]:
                            parsed_value = value.lower() == "true"
                        # Handle numeric values
                        elif value.isdigit():
                            parsed_value = int(value)

                        setattr(config.cache, parts[1], parsed_value)
                    elif parts[0] == "routing" and hasattr(config.routing, parts[1]):
                        setattr(config.routing, parts[1], value)

        return config

    def to_file(self, path: Union[str, Path]) -> None:
        """Save configuration to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump(exclude_none=True)

        with open(path, "w") as f:
            if path.suffix in [".yaml", ".yml"]:
                yaml.safe_dump(data, f, default_flow_style=False)
            elif path.suffix == ".json":
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {path.suffix}")

    def merge(self, other: "Config") -> "Config":
        """Merge another configuration into this one."""

        # Deep merge logic
        def deep_merge(base: dict, update: dict) -> dict:
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base

        merged_data = deep_merge(
            self.model_dump(exclude_none=True),
            other.model_dump(exclude_none=True),
        )

        return Config(**merged_data)

    @classmethod
    def default(cls) -> "Config":
        """Create a default configuration."""
        return cls(
            providers={
                "openai": {
                    "enabled": True,
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "anthropic": {
                    "enabled": True,
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                },
                "google": {
                    "enabled": True,
                    "api_key": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
                },
            },
            routing=RoutingConfig(
                strategy="cost",
                strategy_configs={
                    "cost": {
                        "max_cost_per_1k_tokens": 0.01,
                    },
                    "quality": {
                        "min_quality_tier": "standard",
                    },
                },
            ),
            cache=CacheConfig(
                enabled=True,
                backend="memory",
                ttl=3600,
            ),
            monitoring=MonitoringConfig(
                logging={
                    "level": "INFO",
                    "console_output": True,
                    "rich_formatting": True,
                },
                cost_tracking={
                    "budget_daily": 10.0,
                    "budget_monthly": 100.0,
                },
            ),
        )


def load_config(
    path: Optional[Union[str, Path]] = None,
    use_env: bool = True,
    use_defaults: bool = True,
) -> Config:
    """Load configuration from multiple sources."""
    configs = []

    # Start with defaults if requested
    if use_defaults:
        configs.append(Config.default())

    # Load from environment if requested
    if use_env:
        configs.append(Config.from_env())

    # Load from file if provided
    if path:
        configs.append(Config.from_file(path))

    # Merge all configurations
    if not configs:
        return Config()

    result = configs[0]
    for config in configs[1:]:
        result = result.merge(config)

    return result


def save_config(config: Config, path: Union[str, Path]) -> None:
    """Save configuration to a file."""
    config.to_file(path)
