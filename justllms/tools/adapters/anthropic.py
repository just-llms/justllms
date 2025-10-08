import json
from typing import Any, Dict, List, Optional, Union

from justllms.core.models import Message, Role
from justllms.tools.adapters.base import BaseToolAdapter
from justllms.tools.models import Tool, ToolCall, ToolResult


class AnthropicToolAdapter(BaseToolAdapter):
    """Adapter for Anthropic Claude's tool format.

    Claude uses a different format than OpenAI:
    - Tools have name, description, and input_schema
    - Tool choice is {"type": "auto" | "any" | "tool", "name": "..."}
    - Tool results are sent as user messages with tool_use_id
    """

    def format_tools_for_api(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Convert Tool objects to Anthropic's format.

        Args:
            tools: List of Tool instances.

        Returns:
            List of tool definitions in Anthropic format.

        Example output:
            [
                {
                    "name": "get_weather",
                    "description": "Get weather for location",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name"
                            }
                        },
                        "required": ["location"]
                    }
                }
            ]
        """
        formatted_tools = []

        for tool in tools:
            # Skip native tools (Anthropic doesn't have native tools)
            if tool.is_native:
                continue

            tool_def = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.to_json_schema(),
            }

            formatted_tools.append(tool_def)

        return formatted_tools

    def format_tool_choice(
        self, tool_choice: Optional[Union[str, Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        """Format tool_choice for Anthropic API.

        Args:
            tool_choice: Tool selection strategy:
                - "auto": Let model decide (maps to {"type": "auto"})
                - "none": Don't use tools (not directly supported, omit tools)
                - "required": Must use a tool (maps to {"type": "any"})
                - {"name": "tool_name"}: Use specific tool
                - {"type": "auto|any|tool", "name": "..."}: Full format
                - None: Use default (auto)

        Returns:
            Formatted tool_choice for Anthropic API.
        """
        if tool_choice is None:
            return {"type": "auto"}

        # Handle string choices
        if isinstance(tool_choice, str):
            choice_map = {
                "auto": {"type": "auto"},
                "none": None,  # Anthropic doesn't have explicit "none", handled by not passing tools
                "required": {"type": "any"},  # Map to "any" - must use some tool
            }
            if tool_choice in choice_map:
                return choice_map[tool_choice]
            # Assume it's a tool name
            return {"type": "tool", "name": tool_choice}

        # Handle dict choices
        if isinstance(tool_choice, dict):
            # Check if already in Anthropic format
            if "type" in tool_choice:
                return tool_choice

            # Convert from simple {"name": "..."} format
            if "name" in tool_choice:
                return {"type": "tool", "name": tool_choice["name"]}

        # Default to auto
        return {"type": "auto"}

    def extract_tool_calls(self, response: Dict[str, Any]) -> List[ToolCall]:
        """Extract tool calls from Anthropic response.

        Args:
            response: Raw response from Anthropic API.

        Returns:
            List of ToolCall objects.
        """
        tool_calls = []

        # In Anthropic's format, tool use is in the content array
        content = response.get("content", [])

        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_use":
                # Extract tool call information
                tool_call = ToolCall(
                    id=item.get("id", ""),
                    name=item.get("name", ""),
                    arguments=item.get("input", {}),
                    raw_arguments=json.dumps(item.get("input", {})),
                )
                tool_calls.append(tool_call)

        return tool_calls

    def format_tool_result_message(self, tool_result: ToolResult, tool_call: ToolCall) -> Message:
        """Format tool result as a message for Anthropic.

        Anthropic expects tool results as user messages with tool_result content.

        Args:
            tool_result: Result from tool execution.
            tool_call: Original tool call.

        Returns:
            Message with tool result in Anthropic format.
        """
        # Anthropic uses a content array with tool_result items
        content = [
            {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": tool_result.to_message_content(),
            }
        ]

        return Message(role=Role.USER, content=content)

    def format_tool_calls_message(self, tool_calls: List[ToolCall]) -> Optional[Message]:
        """Format tool calls as an assistant message.

        Anthropic includes tool calls in the assistant's content array.

        Args:
            tool_calls: List of tool calls.

        Returns:
            Assistant message with tool calls in content.
        """
        if not tool_calls:
            return None

        content: List[Dict[str, Any]] = []

        # Add any text content if needed
        # Anthropic requires at least one text block before tool use
        content.append({"type": "text", "text": "I'll help you with that."})

        # Add tool use blocks
        for tc in tool_calls:
            tool_use: Dict[str, Any] = {
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            }
            content.append(tool_use)

        return Message(role=Role.ASSISTANT, content=content)

    def supports_parallel_tools(self) -> bool:
        """Claude 3 supports calling multiple tools in one response."""
        return True

    def supports_required_tools(self) -> bool:
        """Claude supports 'any' which is similar to required."""
        return True

    def get_max_tools_per_call(self) -> Optional[int]:
        """Anthropic doesn't document a specific limit."""
        return None
