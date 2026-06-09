import contextlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from justllms.monitoring.budget import BudgetConfig


class ConfigProviderSettings(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    api_key: Optional[str] = None
    enabled: bool = True
    base_url: Optional[str] = None
    timeout: Optional[int] = None
    max_retries: int = 3
    rate_limit: Optional[int] = None
    deployment_mapping: Dict[str, str] = Field(default_factory=dict)


class RoutingConfig(BaseModel):
    """Configuration for provider and model fallbacks."""

    model_config = ConfigDict(extra="allow")

    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None

    """Ordered failover chain. Entries are "provider/model" or bare "provider"
    (which resolves to the provider's first available model)."""
    fallback_chain: List[str] = Field(default_factory=list)

    """When True and fallback_chain is empty, derive a chain at request time
    from all other configured providers (their first available model)."""
    auto_fallback: bool = False

    """Total attempts including the primary provider."""
    max_fallback_attempts: int = 3

    """Error categories that trigger failover to the next provider."""
    fallback_on: List[str] = Field(
        default_factory=lambda: ["rate_limit", "server_error", "timeout", "connection", "auth"]
    )

    """max execution time per tool"""
    tool_timeout: float = 120.0

    """max number of tool execution rounds"""
    max_tool_iterations: int = 10

    """whether to automatically execute tools by default"""
    execute_tools_by_default: bool = False


class CacheConfig(BaseModel):
    """Configuration for response caching."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    """Whether response caching is enabled."""

    backend: str = "memory"
    """Cache backend to use: "memory" or "disk"."""

    ttl: Optional[float] = 3600.0
    """Default time-to-live in seconds for cached entries. None = no expiry."""

    max_size: int = 1000
    """Maximum number of entries (memory backend only)."""

    path: Optional[str] = None
    """Database file path for the disk backend (default ~/.justllms/cache.db)."""


class Config(BaseModel):
    """Configuration class for multi-provider LLM client."""

    model_config = ConfigDict(extra="allow")

    providers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    budget: Optional[BudgetConfig] = None
    cache: CacheConfig = Field(default_factory=CacheConfig)

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

        if data is None:
            data = {}
        elif not isinstance(data, dict):
            raise ValueError(
                f"Configuration file must contain a mapping at the top level, got {type(data).__name__}"
            )

        return cls(**data)

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        providers = {}

        # Common provider environment variables
        provider_keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "azure_openai": "AZURE_OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "grok": ("XAI_API_KEY", "GROK_API_KEY"),  # Support both for backwards compatibility
        }

        for provider_name, env_key in provider_keys.items():
            if isinstance(env_key, tuple):
                api_key = None
                for key in env_key:
                    api_key = os.getenv(key)
                    if api_key:
                        break
            else:
                # Type guard: env_key is a string here
                api_key = os.getenv(env_key)  # type: ignore[call-overload]

            if api_key:
                providers[provider_name] = {"api_key": api_key}

        ollama_base = os.getenv("OLLAMA_API_BASE") or os.getenv("OLLAMA_HOST")
        ollama_enabled = os.getenv("OLLAMA_ENABLED", "").lower() in {"1", "true", "yes"}
        if ollama_base or ollama_enabled:
            provider_entry: Dict[str, Any] = {"enabled": True}
            if ollama_base:
                provider_entry["base_url"] = ollama_base

            headers_json = os.getenv("OLLAMA_HEADERS_JSON")
            if headers_json:
                with contextlib.suppress(json.JSONDecodeError):
                    provider_entry["headers"] = json.loads(headers_json)

            providers["ollama"] = provider_entry

        return cls(providers=providers, routing=RoutingConfig())


def load_config(
    config_path: Optional[str] = None,
    use_defaults: bool = True,
    use_env: bool = True,
) -> Config:
    """Load configuration from various sources."""
    if config_path:
        return Config.from_file(config_path)

    # Try to find config file
    config_files = ["justllms.yaml", "justllms.yml", "justllms.json"]
    for config_file in config_files:
        if Path(config_file).exists():
            return Config.from_file(config_file)

    if use_env:
        return Config.from_env()

    if use_defaults:
        return Config()

    raise FileNotFoundError("No configuration file found and environment variables not available")
