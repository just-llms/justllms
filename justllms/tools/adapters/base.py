from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from justllms.core.models import Message, Role
from justllms.tools.models import Tool, ToolCall, ToolResult


class BaseToolAdapter(ABC):
    """Abstract base class for provider-specific tool format conversion.

    Each provider (OpenAI, Anthropic, Google, etc.) has its own format
    for tool/function definitions and tool call responses. This adapter
    provides a unified interface for conversion between JustLLMs Tool
    objects and provider-specific formats.
    """

    @abstractmethod
    def format_tools_for_api(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Convert Tool objects to provider's API format.

        Args:
            tools: List of Tool instances to convert.

        Returns:
            List of tool definitions in provider-specific format.

        Examples:
            OpenAI format:
                [{"type": "function", "function": {...}}]

            Anthropic format:
                [{"name": "...", "description": "...", "input_schema": {...}}]
        """
        pass

    @abstractmethod
    def format_tool_choice(
        self, tool_choice: Optional[Union[str, Dict[str, Any]]]
    ) -> Optional[Any]:
        """Normalize tool_choice parameter for provider.

        Args:
            tool_choice: Tool selection strategy. Can be:
                - "auto": Let model decide
                - "none": Don't use tools
                - "required": Must use a tool (OpenAI)
                - Dict with specific tool name
                - None: Use provider default

        Returns:
            Provider-specific tool_choice format.
        """
        pass

    @abstractmethod
    def extract_tool_calls(self, response: Dict[str, Any]) -> List[ToolCall]:
        """Extract tool calls from provider's response.

        Args:
            response: Raw response from provider API.

        Returns:
            List of ToolCall objects extracted from response.
        """
        pass

    @abstractmethod
    def format_tool_result_message(self, tool_result: ToolResult, tool_call: ToolCall) -> Message:
        """Format tool execution result as a message.

        Args:
            tool_result: Result from tool execution.
            tool_call: Original tool call that produced this result.

        Returns:
            Message object formatted for the provider.
        """
        pass

    def format_tool_calls_message(self, tool_calls: List[ToolCall]) -> Optional[Message]:
        """Format tool calls as an assistant message.

        Some providers need tool calls formatted as assistant messages.

        Args:
            tool_calls: List of tool calls to include in message.

        Returns:
            Assistant message with tool calls, or None if not needed.
        """
        # Default implementation for OpenAI-style
        if not tool_calls:
            return None

        tool_calls_data = []
        for tc in tool_calls:
            tool_call_dict = {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.raw_arguments or self._serialize_arguments(tc.arguments),
                },
            }
            tool_calls_data.append(tool_call_dict)

        return Message(
            role=Role.ASSISTANT,
            content="",  # Tool calls don't have content
            tool_calls=tool_calls_data,
        )

    def supports_parallel_tools(self) -> bool:
        """Check if provider supports parallel tool calls.

        Returns:
            True if provider can call multiple tools in one response.
        """
        return False

    def supports_required_tools(self) -> bool:
        """Check if provider supports 'required' tool choice.

        Returns:
            True if provider supports forcing tool use.
        """
        return False

    def get_max_tools_per_call(self) -> Optional[int]:
        """Get maximum number of tools that can be defined per call.

        Returns:
            Maximum number or None if unlimited.
        """
        return None

    def _serialize_arguments(self, arguments: Dict[str, Any]) -> str:
        """Serialize arguments dictionary to JSON string.

        Args:
            arguments: Tool arguments dictionary.

        Returns:
            JSON string representation.
        """
        import json

        return json.dumps(arguments, default=str)

    def _parse_arguments(self, arguments_str: str) -> Dict[str, Any]:
        """Parse JSON string to arguments dictionary.

        Args:
            arguments_str: JSON string of arguments.

        Returns:
            Parsed arguments dictionary.
        """
        import json
        from typing import cast

        try:
            return cast(Dict[str, Any], json.loads(arguments_str))
        except json.JSONDecodeError:
            # Return as-is if not valid JSON
            return {"raw": arguments_str}
