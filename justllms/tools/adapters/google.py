import json
from typing import Any, Dict, List, Optional, Union

from justllms.core.models import Message, Role
from justllms.tools.adapters.base import BaseToolAdapter
from justllms.tools.models import Tool, ToolCall, ToolResult


class GoogleToolAdapter(BaseToolAdapter):
    """Adapter for Google Gemini's function calling format.

    Gemini uses yet another format:
    - Functions are defined in a tools array with functionDeclarations
    - Tool choice uses toolConfig with functionCallingConfig
    - Function calls come back in functionCall objects
    - Gemini also supports native tools like google_search
    """

    def format_tools_for_api(
        self, tools: List[Tool], include_native: bool = True
    ) -> List[Dict[str, Any]]:
        """Convert Tool objects to Gemini's function format.

        Args:
            tools: List of Tool instances (user-defined and/or native).
            include_native: Whether to include native Google tools in API format.

        Returns:
            List containing tool configuration for Gemini.

        Example output:
            [
                # Native tools in their special format
                {"googleSearch": {}},
                {"codeExecution": {}},
                # User tools in function declarations
                {
                    "functionDeclarations": [
                        {
                            "name": "get_weather",
                            "description": "Get weather for location",
                            "parameters": {
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
                }
            ]
        """
        api_tools = []
        function_declarations = []

        for tool in tools:
            # Handle native Google tools - they use a different format
            if tool.is_native and tool.namespace == "google" and include_native:
                if hasattr(tool, "to_api_format"):
                    # Native tools have their own API format method
                    api_tools.append(tool.to_api_format())
                continue

            # Regular user-defined tools
            function_def = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.to_json_schema(),
            }

            function_declarations.append(function_def)

        # Add user tools as function declarations
        if function_declarations:
            api_tools.append({"functionDeclarations": function_declarations})

        return api_tools

    def format_tools_with_native(
        self, user_tools: List[Tool], native_tools: List[Any]
    ) -> List[Dict[str, Any]]:
        """Format both user-defined and native tools for Gemini API.

        This method handles the new importable API where users pass native tools
        directly: tools=[GoogleSearch(), multiply]

        Args:
            user_tools: List of user-defined Tool objects.
            native_tools: List of native tool objects (e.g., GoogleSearch, GoogleCodeExecution).

        Returns:
            List of tool configurations in Gemini format:
            [
                {"google_search": {}},
                {"code_execution": {}},
                {"function_declarations": [...]}
            ]

        Note: When mixing native tools with user tools, we use "function_declarations"
        (snake_case) as per Gemini's live-tools documentation.
        """
        api_tools = []

        # Add native tools first (they have their own to_api_format method)
        for native_tool in native_tools:
            if hasattr(native_tool, "to_api_format"):
                api_tools.append(native_tool.to_api_format())

        # Add user-defined tools as function declarations
        # When mixing with native tools, use snake_case key as per Gemini live-tools docs
        if user_tools:
            function_declarations = []
            for tool in user_tools:
                function_def = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.to_json_schema(),
                }
                function_declarations.append(function_def)

            # Use snake_case when there are native tools, camelCase otherwise
            key = "function_declarations" if native_tools else "functionDeclarations"
            api_tools.append({key: function_declarations})

        return api_tools

    def format_tool_choice(
        self, tool_choice: Optional[Union[str, Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        """Format tool_choice as toolConfig for Gemini API.

        Args:
            tool_choice: Tool selection strategy:
                - "auto": Let model decide
                - "none": Don't use functions
                - "required": Must call a function (ANY mode)
                - {"name": "function_name"}: Not directly supported
                - None: Use default (auto)

        Returns:
            toolConfig dict for Gemini API.
        """
        if tool_choice is None or tool_choice == "auto":
            # AUTO mode - model decides
            return {"functionCallingConfig": {"mode": "AUTO"}}

        if tool_choice == "none":
            # NONE mode - no function calls
            return {"functionCallingConfig": {"mode": "NONE"}}

        if tool_choice == "required":
            # ANY mode - must call at least one function
            return {"functionCallingConfig": {"mode": "ANY"}}

        # Gemini doesn't support forcing specific functions directly
        # Default to AUTO
        return {"functionCallingConfig": {"mode": "AUTO"}}

    def extract_tool_calls(self, response: Dict[str, Any]) -> List[ToolCall]:
        """Extract tool calls from Gemini response.

        Args:
            response: Raw response from Gemini API.

        Returns:
            List of ToolCall objects.
        """
        tool_calls = []

        # Navigate Gemini's response structure
        candidates = response.get("candidates", [])
        for candidate in candidates:
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            for part in parts:
                # Check for function calls
                if "functionCall" in part:
                    function_call = part["functionCall"]

                    # Extract arguments
                    args = function_call.get("args", {})

                    tool_call = ToolCall(
                        id=f"call_{function_call.get('name', 'unknown')}",
                        name=function_call.get("name", ""),
                        arguments=args,
                        raw_arguments=json.dumps(args),
                    )
                    tool_calls.append(tool_call)

        return tool_calls

    def format_tool_result_message(self, tool_result: ToolResult, tool_call: ToolCall) -> Message:
        """Format tool result as a message for Gemini.

        Gemini expects function responses in a specific format.

        Args:
            tool_result: Result from tool execution.
            tool_call: Original tool call.

        Returns:
            Message with function response for Gemini.
        """
        # Gemini expects function responses as parts
        # Pass as list directly so GoogleProvider._format_messages can use it
        parts = [
            {
                "functionResponse": {
                    "name": tool_call.name,
                    "response": {"result": tool_result.to_message_content()},
                }
            }
        ]

        # Gemini uses USER role for function responses
        # Pass parts list directly, not as JSON string
        return Message(
            role=Role.USER,
            content=parts,  # Pass as list for GoogleProvider to handle
        )

    def format_tool_calls_message(self, tool_calls: List[ToolCall]) -> Optional[Message]:
        """Format tool calls as an assistant message for Gemini.

        Args:
            tool_calls: List of tool calls.

        Returns:
            Assistant message with function calls.
        """
        if not tool_calls:
            return None

        parts = []
        for tc in tool_calls:
            part = {"functionCall": {"name": tc.name, "args": tc.arguments}}
            parts.append(part)

        # Pass parts list directly for GoogleProvider to handle
        return Message(
            role=Role.ASSISTANT,
            content=parts,  # Pass as list, not JSON string
        )

    def merge_native_tools(
        self, user_tools: List[Tool], native_config: Optional[Dict[str, Any]]
    ) -> List[Tool]:
        """Merge user-defined tools with Google's native tools.

        Args:
            user_tools: User-defined Tool instances.
            native_config: Configuration for native tools from provider config.

        Returns:
            Combined list of tools (native tools + user tools).

        Example:
            native_config = {
                "google_search": {"enabled": True},
                "code_execution": {"enabled": True}
            }
            merged = adapter.merge_native_tools(user_tools, native_config)
        """
        if not native_config:
            return user_tools

        # Use NativeToolManager to load and merge tools
        from justllms.tools.native.manager import GoogleNativeToolManager

        manager = GoogleNativeToolManager(config=native_config)

        # Merge with prefer_native=True so native tools take precedence
        merged_tools = manager.merge_with_user_tools(user_tools, prefer_native=True)

        return merged_tools

    def supports_parallel_tools(self) -> bool:
        """Gemini supports calling multiple functions in one response."""
        return True

    def supports_required_tools(self) -> bool:
        """Gemini supports ANY mode which is like required."""
        return True

    def get_max_tools_per_call(self) -> Optional[int]:
        """Gemini supports up to 64 function declarations."""
        return 64
