from typing import Dict, List, Optional

from justllms.tools.models import Tool


class ToolRegistry:
    """Manages a collection of tools with namespace support.

    The registry provides a way to organize and access tools,
    with optional namespace isolation.

    Attributes:
        namespace: Optional namespace for this registry.
        _tools: Dictionary mapping tool names to Tool instances.

    Examples:
        >>> registry = ToolRegistry(namespace="math")
        >>> registry.register(add_tool)
        >>> registry.register(multiply_tool)
        >>> print(registry.list_tools())
        ['add', 'multiply']

        >>> # Get specific tool
        >>> tool = registry.get_tool("add")

        >>> # Merge registries
        >>> other_registry = ToolRegistry(namespace="text")
        >>> combined = registry.merge(other_registry)
    """

    def __init__(self, namespace: Optional[str] = None):
        """Initialize a new tool registry.

        Args:
            namespace: Optional namespace for tools in this registry.
        """
        self.namespace = namespace
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool in the registry.

        If the tool doesn't have a namespace and the registry does,
        the registry's namespace will be applied to the tool.

        Args:
            tool: The Tool instance to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        # Apply registry namespace if tool doesn't have one
        if self.namespace and not tool.namespace:
            tool.namespace = self.namespace

        if tool.name in self._tools:
            existing = self._tools[tool.name]
            raise ValueError(
                f"Tool '{tool.name}' is already registered in this registry. "
                f"Existing tool: {existing.description}"
            )

        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry.

        Args:
            name: Name of the tool to remove.

        Raises:
            KeyError: If the tool doesn't exist.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        del self._tools[name]

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name.

        Args:
            name: Name of the tool to retrieve.

        Returns:
            Tool instance if found, None otherwise.
        """
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all tool names in the registry.

        Returns:
            List of tool names.
        """
        return list(self._tools.keys())

    def get_all_tools(self) -> List[Tool]:
        """Get all Tool instances in the registry.

        Returns:
            List of all Tool instances.
        """
        return list(self._tools.values())

    def merge(self, other: "ToolRegistry", check_conflicts: bool = True) -> "ToolRegistry":
        """Merge another registry into a new registry.

        Creates a new registry containing tools from both registries.
        Does not modify either original registry.

        Args:
            other: Another ToolRegistry to merge with.
            check_conflicts: Whether to check for name conflicts.

        Returns:
            New ToolRegistry containing tools from both.

        Raises:
            ValueError: If check_conflicts is True and there are name conflicts.
        """
        # Create new registry with no specific namespace
        merged = ToolRegistry()

        # Add tools from this registry
        for tool in self._tools.values():
            merged._tools[tool.name] = tool

        # Add tools from other registry
        for tool in other._tools.values():
            if check_conflicts and tool.name in merged._tools:
                existing = merged._tools[tool.name]
                if existing.full_name != tool.full_name:
                    raise ValueError(
                        f"Tool name conflict: '{tool.name}' exists in both registries. "
                        f"Existing: {existing.full_name}, New: {tool.full_name}"
                    )
            merged._tools[tool.name] = tool

        return merged

    def clear(self) -> None:
        """Remove all tools from the registry."""
        self._tools.clear()

    def __len__(self) -> int:
        """Return the number of tools in the registry."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool exists in the registry.

        Args:
            name: Name of the tool to check.

        Returns:
            True if the tool exists, False otherwise.
        """
        return name in self._tools

    def __repr__(self) -> str:
        """String representation of the registry."""
        namespace_str = f", namespace='{self.namespace}'" if self.namespace else ""
        return f"ToolRegistry(tools={len(self._tools)}{namespace_str})"


class GlobalToolRegistry(ToolRegistry):
    """Singleton global tool registry.

    This registry is shared across the application and can be used
    to register tools that should be available globally.

    Examples:
        >>> from justllms.tools.registry import GlobalToolRegistry
        >>> registry = GlobalToolRegistry()
        >>> registry.register(my_tool)
        >>> # Access from anywhere
        >>> registry2 = GlobalToolRegistry()
        >>> assert registry is registry2  # Same instance
    """

    _instance: Optional["GlobalToolRegistry"] = None

    def __new__(cls) -> "GlobalToolRegistry":
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the global registry."""
        # Only initialize once
        if not getattr(self, "_GlobalToolRegistry__initialized", False):
            super().__init__(namespace=None)
            self.__initialized = True

    @classmethod
    def reset(cls) -> None:
        """Reset the global registry (mainly for testing)."""
        if cls._instance:
            cls._instance.clear()
            cls._instance = None
