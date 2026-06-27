import json
from typing import Any, Dict, List, Optional, Union

from justllms.core.models import Message, Role
from justllms.tools.adapters.base import BaseToolAdapter
from justllms.tools.models import Tool, ToolCall, ToolResult


class OllamaToolAdapter(BaseToolAdapter):
    """Adapter for Ollama's ``/api/chat`` tool calling format.

    Ollama accepts OpenAI-style tool definitions, but its responses differ:

    - tool-call ``arguments`` arrive as a JSON object, not a JSON string,
    - tool calls do not carry an ``id``, and
    - tool results are returned with ``role: "tool"`` and a ``tool_name`` field
      (there is no ``tool_call_id``).
    """

    def format_tools_for_api(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Convert Tool objects to Ollama's (OpenAI-style) function format."""
        formatted_tools = []

        for tool in tools:
            if tool.is_native:
                continue

            formatted_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.to_json_schema(),
                    },
                }
            )

        return formatted_tools

    def format_tool_choice(
        self, tool_choice: Optional[Union[str, Dict[str, Any]]]
    ) -> Optional[Any]:
        """Ollama's chat API has no tool_choice field; the model decides.

        Returning None keeps it out of the request payload.
        """
        return None

    def extract_tool_calls(self, response: Dict[str, Any]) -> List[ToolCall]:
        """Extract tool calls from an Ollama response.

        Handles arguments delivered as a dict (the Ollama default) as well as a
        JSON string, and synthesizes a stable id since Ollama omits one.
        """
        tool_calls = []

        for choice_idx, choice in enumerate(response.get("choices", [])):
            message = choice.get("message", {})

            for tc_idx, tc in enumerate(message.get("tool_calls") or []):
                function = tc.get("function", {})

                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}

                name = function.get("name", "")
                call_id = tc.get("id") or f"call_{choice_idx}_{tc_idx}_{name}"

                tool_calls.append(
                    ToolCall(
                        id=call_id,
                        name=name,
                        arguments=arguments,
                        raw_arguments=json.dumps(arguments),
                    )
                )

        return tool_calls

    def format_tool_calls_message(self, tool_calls: List[ToolCall]) -> Optional[Message]:
        """Format tool calls as an assistant message in Ollama's format.

        Ollama expects ``function.arguments`` as an object (not a string) and
        does not use ``id``/``type`` on echoed calls.
        """
        if not tool_calls:
            return None

        tool_calls_data = [
            {"type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
            for tc in tool_calls
        ]

        return Message(role=Role.ASSISTANT, content="", tool_calls=tool_calls_data)

    def format_tool_result_message(self, tool_result: ToolResult, tool_call: ToolCall) -> Message:
        """Format a tool result as an Ollama ``role: "tool"`` message.

        The result reference is carried via ``tool_name`` (Ollama's field),
        which the provider serializes into the request.
        """
        return Message(
            role=Role.TOOL,
            content=tool_result.to_message_content(),
            tool_name=tool_call.name,
        )

    def supports_parallel_tools(self) -> bool:
        """Ollama can return multiple tool calls in one response."""
        return True

    def get_max_tools_per_call(self) -> Optional[int]:
        """Ollama does not document a specific limit."""
        return None
