from typing import Any, Dict, List, Optional

from justllms.tools.models import Tool
from justllms.tools.native.google_tools import GOOGLE_NATIVE_TOOLS, get_google_native_tool


class NativeToolManager:
    """Manages native tools for providers.

    Responsibilities:
    - Discover enabled native tools from provider config
    - Instantiate native tool objects with their configs
    - Merge native tools with user-defined tools
    - Handle namespace conflicts
    """

    def __init__(self, provider: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the native tool manager.

        Args:
            provider: Provider name (e.g., "google", "openai").
            config: Provider's native_tools configuration dict.
        """
        self.provider = provider
        self.config = config or {}
        self._native_tools: Dict[str, Tool] = {}
        self._load_native_tools()

    def _load_native_tools(self) -> None:
        """Load and instantiate enabled native tools based on config."""
        if self.provider == "google":
            self._load_google_tools()
        # Add more providers as they support native tools
        # elif self.provider == "openai":
        #     self._load_openai_tools()

    def _load_google_tools(self) -> None:
        """Load Google native tools from config.

        Config format:
            {
                "google_search": {
                    "enabled": True,
                    "dynamic_retrieval_config": {...}
                },
                "code_execution": {
                    "enabled": True,
                    "timeout": 30
                }
            }
        """
        for tool_name, tool_config in self.config.items():
            # Skip if not enabled
            if not tool_config.get("enabled", False):
                continue

            # Check if it's a valid Google native tool
            if tool_name not in GOOGLE_NATIVE_TOOLS:
                continue

            try:
                # Instantiate the tool with its config
                tool_instance = get_google_native_tool(tool_name, tool_config)
                self._native_tools[tool_name] = tool_instance
            except Exception:
                # Silently skip tools that fail to initialize
                pass

    def get_native_tools(self) -> List[Tool]:
        """Get all loaded native tools.

        Returns:
            List of native Tool instances.
        """
        return list(self._native_tools.values())

    def get_native_tool(self, name: str) -> Optional[Tool]:
        """Get a specific native tool by name.

        Args:
            name: Tool name.

        Returns:
            Tool instance if found, None otherwise.
        """
        return self._native_tools.get(name)

    def has_native_tool(self, name: str) -> bool:
        """Check if a native tool is loaded.

        Args:
            name: Tool name.

        Returns:
            True if tool is loaded, False otherwise.
        """
        return name in self._native_tools

    def merge_with_user_tools(
        self, user_tools: List[Tool], prefer_native: bool = True
    ) -> List[Tool]:
        """Merge native tools with user-defined tools.

        Args:
            user_tools: User-defined tools.
            prefer_native: If True, native tools override user tools with same name.
                          If False, user tools take precedence.

        Returns:
            Combined list of tools.
        """
        # Create a dict to track tools by qualified name (namespace:name)
        merged: Dict[str, Tool] = {}

        # Add native tools first if prefer_native, else user tools first
        first_tools = self._native_tools.values() if prefer_native else user_tools
        second_tools = user_tools if prefer_native else self._native_tools.values()

        # Add first set of tools
        for tool in first_tools:
            key = f"{tool.namespace}:{tool.name}" if tool.namespace else tool.name
            merged[key] = tool

        # Add second set, respecting the preference order
        for tool in second_tools:
            key = f"{tool.namespace}:{tool.name}" if tool.namespace else tool.name
            if key not in merged:
                merged[key] = tool

        return list(merged.values())

    def get_api_format_for_google(self) -> List[Dict[str, Any]]:
        """Get Google-specific API format for native tools.

        Google native tools use a different format than regular function declarations.
        They're added to the "tools" array with special keys like "googleSearch" or
        "codeExecution".

        Returns:
            List of native tool dicts in Google API format.
        """
        api_tools = []

        for tool in self._native_tools.values():
            if hasattr(tool, "to_api_format"):
                api_tools.append(tool.to_api_format())

        return api_tools

    def __len__(self) -> int:
        """Return number of loaded native tools."""
        return len(self._native_tools)

    def __bool__(self) -> bool:
        """Return True if any native tools are loaded."""
        return len(self._native_tools) > 0


class GoogleNativeToolManager(NativeToolManager):
    """Specialized manager for Google native tools.

    Provides Google-specific functionality for native tool management.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Google native tool manager.

        Args:
            config: Google native_tools configuration.
        """
        super().__init__(provider="google", config=config)

    def format_for_gemini_api(
        self, user_tools: Optional[List[Tool]] = None
    ) -> List[Dict[str, Any]]:
        """Format tools for Gemini API request.

        Gemini requires native tools and function declarations in separate formats
        within the same "tools" array.

        Args:
            user_tools: Optional user-defined tools to include.

        Returns:
            List of formatted tool definitions for Gemini API.
        """
        tools_array: List[Dict[str, Any]] = []

        # Add native tools in their special format
        native_tools_api = self.get_api_format_for_google()
        if native_tools_api:
            # Native tools go directly in the tools array
            tools_array.extend(native_tools_api)

        # Add user-defined tools as function declarations
        if user_tools:
            from justllms.tools.adapters.google import GoogleToolAdapter

            adapter = GoogleToolAdapter()
            user_tools_formatted = adapter.format_tools_for_api(user_tools)

            # User tools are wrapped in functionDeclarations
            if user_tools_formatted:
                tools_array.extend(user_tools_formatted)

        return tools_array


def create_native_tool_manager(
    provider: str, config: Optional[Dict[str, Any]] = None
) -> Optional[NativeToolManager]:
    """Factory function to create the appropriate native tool manager.

    Args:
        provider: Provider name (e.g., "google", "openai").
        config: Provider's native_tools configuration.

    Returns:
        NativeToolManager instance if provider supports native tools, None otherwise.
    """
    if provider == "google":
        return GoogleNativeToolManager(config=config)

    # Add more providers as they support native tools
    # elif provider == "openai":
    #     return OpenAINativeToolManager(config=config)

    return None
