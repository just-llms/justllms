import json
from typing import Any, Dict, List, Optional, Union

from justllms.core.models import Message, Role
from justllms.tools.adapters.base import BaseToolAdapter
from justllms.tools.models import Tool, ToolCall, ToolResult


class OpenAIToolAdapter(BaseToolAdapter):
    """Adapter for OpenAI's function calling format.

    OpenAI uses a specific format for function definitions and tool calls:
    - Functions are wrapped in {"type": "function", "function": {...}}
    - Tool choice can be "auto", "none", "required", or specific function
    - Tool calls are returned in the response choices

    This adapter is also used by Azure OpenAI since they share the same API.
    """

    def format_tools_for_api(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Convert Tool objects to OpenAI's function format.

        Args:
            tools: List of Tool instances.

        Returns:
            List of function definitions in OpenAI format.

        Example output:
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                            },
                            "required": ["location"]
                        }
                    }
                }
            ]
        """
        formatted_tools = []

        for tool in tools:
            # Skip native tools (OpenAI doesn't have native tools)
            if tool.is_native:
                continue

            function_def = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.to_json_schema(),
                },
            }

            formatted_tools.append(function_def)

        return formatted_tools

    def format_tool_choice(
        self, tool_choice: Optional[Union[str, Dict[str, Any]]]
    ) -> Optional[Union[str, Dict[str, Any]]]:
        """Format tool_choice for OpenAI API.

        Args:
            tool_choice: Tool selection strategy:
                - "auto": Model decides whether to use tools
                - "none": Model will not call functions
                - "required": Model must call at least one function
                - {"name": "function_name"}: Force specific function
                - {"type": "function", "function": {"name": "..."}}: Full format
                - None: Use default (auto)

        Returns:
            Formatted tool_choice for OpenAI API.
        """
        if tool_choice is None:
            return "auto"

        # Handle string choices
        if isinstance(tool_choice, str):
            if tool_choice in ("auto", "none", "required"):
                return tool_choice
            # If it's a function name as string, convert to dict
            return {"type": "function", "function": {"name": tool_choice}}

        # Handle dict choices
        if isinstance(tool_choice, dict):
            # Check if it's already in full format
            if "type" in tool_choice and tool_choice["type"] == "function":
                return tool_choice

            # Check if it's simplified format {"name": "function_name"}
            if "name" in tool_choice:
                return {"type": "function", "function": {"name": tool_choice["name"]}}

        # Default to auto if format not recognized
        return "auto"

    def extract_tool_calls(self, response: Dict[str, Any]) -> List[ToolCall]:
        """Extract tool calls from OpenAI response.

        Args:
            response: Raw response from OpenAI API.

        Returns:
            List of ToolCall objects.
        """
        tool_calls = []

        # Check choices for tool calls
        choices = response.get("choices", [])
        for choice in choices:
            message = choice.get("message", {})

            # Check for tool_calls in message
            message_tool_calls = message.get("tool_calls", [])
            for tc in message_tool_calls:
                function = tc.get("function", {})

                # Parse arguments
                arguments_str = function.get("arguments", "{}")
                try:
                    arguments = json.loads(arguments_str)
                except json.JSONDecodeError:
                    arguments = {}

                tool_call = ToolCall(
                    id=tc.get("id", ""),
                    name=function.get("name", ""),
                    arguments=arguments,
                    raw_arguments=arguments_str,
                )
                tool_calls.append(tool_call)

            # Also check for legacy function_call format
            function_call = message.get("function_call")
            if function_call:
                arguments_str = function_call.get("arguments", "{}")
                try:
                    arguments = json.loads(arguments_str)
                except json.JSONDecodeError:
                    arguments = {}

                tool_call = ToolCall(
                    id=f"call_{function_call.get('name', 'unknown')}",
                    name=function_call.get("name", ""),
                    arguments=arguments,
                    raw_arguments=arguments_str,
                )
                tool_calls.append(tool_call)

        return tool_calls

    def format_tool_result_message(self, tool_result: ToolResult, tool_call: ToolCall) -> Message:
        """Format tool result as a message for OpenAI.

        OpenAI expects tool results as messages with role="tool".

        Args:
            tool_result: Result from tool execution.
            tool_call: Original tool call.

        Returns:
            Message with role="tool" containing the result.
        """
        content = tool_result.to_message_content()

        return Message(
            role=Role.TOOL,
            content=content,
            name=tool_call.name,
            tool_call_id=tool_call.id,  # OpenAI requires matching the tool call ID
        )

    def format_tool_calls_message(self, tool_calls: List[ToolCall]) -> Optional[Message]:
        """Format tool calls as an assistant message.

        OpenAI requires tool calls to be in an assistant message.

        Args:
            tool_calls: List of tool calls.

        Returns:
            Assistant message with tool calls.
        """
        if not tool_calls:
            return None

        tool_calls_data = []
        for tc in tool_calls:
            tool_call_dict = {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.raw_arguments or json.dumps(tc.arguments),
                },
            }
            tool_calls_data.append(tool_call_dict)

        return Message(
            role=Role.ASSISTANT,
            content="",  # OpenAI tool calls have empty content
            tool_calls=tool_calls_data,
        )

    def supports_parallel_tools(self) -> bool:
        """OpenAI supports calling multiple tools in parallel."""
        return True

    def supports_required_tools(self) -> bool:
        """OpenAI supports the 'required' tool_choice option."""
        return True

    def get_max_tools_per_call(self) -> Optional[int]:
        """OpenAI has a limit of 128 functions per request."""
        return 128
