import contextlib
from typing import Any, Dict, List, Optional

from justllms.tools.models import Tool
from justllms.tools.native.google_tools import GOOGLE_NATIVE_TOOLS, get_google_native_tool


class GoogleNativeToolManager:
    """Manages Google native tools loaded from provider config."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._native_tools: Dict[str, Tool] = {}
        self._load_google_tools()

    def _load_google_tools(self) -> None:
        for tool_name, tool_config in self.config.items():
            if not tool_config.get("enabled", False):
                continue
            if tool_name not in GOOGLE_NATIVE_TOOLS:
                continue
            with contextlib.suppress(Exception):
                self._native_tools[tool_name] = get_google_native_tool(tool_name, tool_config)

    def get_native_tools(self) -> List[Tool]:
        return list(self._native_tools.values())

    def merge_with_user_tools(
        self, user_tools: List[Tool], prefer_native: bool = True
    ) -> List[Tool]:
        merged: Dict[str, Tool] = {}
        first = self._native_tools.values() if prefer_native else user_tools
        second = user_tools if prefer_native else self._native_tools.values()
        for tool in first:
            merged[f"{tool.namespace}:{tool.name}" if tool.namespace else tool.name] = tool
        for tool in second:
            key = f"{tool.namespace}:{tool.name}" if tool.namespace else tool.name
            if key not in merged:
                merged[key] = tool
        return list(merged.values())

    def get_api_format_for_google(self) -> List[Dict[str, Any]]:
        return [tool.to_api_format() for tool in self._native_tools.values() if hasattr(tool, "to_api_format")]
