import json
import threading
import time
from typing import Any, Dict, List, Optional

from justllms.core.base import BaseResponse
from justllms.tools.models import Tool, ToolCall, ToolExecutionEntry, ToolResult, ToolResultStatus
from justllms.tools.utils import validate_tool_arguments


class ToolExecutor:
    """Executes tools sequentially with error handling.

    This executor handles tool validation, argument parsing, execution,
    and error recovery. It does NOT support parallel execution - all
    tools run sequentially.

    Attributes:
        tools: Dictionary mapping tool names to Tool instances.
        timeout: Maximum execution time per tool in seconds.
        execute_in_parallel: Always False (no parallel execution).
    """

    def __init__(
        self,
        tools: List[Tool],
        execute_in_parallel: bool = False,
        timeout: float = 30.0,
    ):
        """Initialize the tool executor.

        Args:
            tools: List of Tool instances available for execution.
            execute_in_parallel: Ignored (always False, no parallel support).
            timeout: Maximum execution time per tool in seconds.
        """
        self.tools = {tool.name: tool for tool in tools}
        self.execute_in_parallel = False  # Always False per requirements
        self.timeout = timeout
        self._execution_count = 0

    def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call with timeout and error handling.

        Args:
            tool_call: The tool call to execute.

        Returns:
            ToolResult with execution outcome.
        """
        start_time = time.time()

        # Find the tool
        tool = self.tools.get(tool_call.name)
        if not tool:
            return ToolResult(
                tool_call_id=tool_call.id,
                result=None,
                error=f"Tool '{tool_call.name}' not found",
                execution_time_ms=(time.time() - start_time) * 1000,
                status=ToolResultStatus.ERROR,
            )

        # Validate and prepare arguments
        try:
            validated_args = validate_tool_arguments(tool, tool_call.arguments)
        except ValueError as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                result=None,
                error=f"Invalid arguments: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
                status=ToolResultStatus.ERROR,
            )

        # Execute with timeout
        result_container: Dict[str, Any] = {}

        def execute_tool() -> None:
            """Execute tool in separate thread for timeout control."""
            try:
                result_container["result"] = tool.callable(**validated_args)
                result_container["success"] = True
            except Exception as e:
                result_container["error"] = str(e)
                result_container["success"] = False

        # Run with timeout
        thread = threading.Thread(target=execute_tool, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)

        execution_time_ms = (time.time() - start_time) * 1000

        # Check timeout
        if thread.is_alive():
            return ToolResult(
                tool_call_id=tool_call.id,
                result=None,
                error=f"Tool execution timed out after {self.timeout}s",
                execution_time_ms=execution_time_ms,
                status=ToolResultStatus.TIMEOUT,
            )

        # Check for errors
        if not result_container.get("success", False):
            return ToolResult(
                tool_call_id=tool_call.id,
                result=None,
                error=result_container.get("error", "Unknown error"),
                execution_time_ms=execution_time_ms,
                status=ToolResultStatus.ERROR,
            )

        # Success
        return ToolResult(
            tool_call_id=tool_call.id,
            result=result_container.get("result"),
            error=None,
            execution_time_ms=execution_time_ms,
            status=ToolResultStatus.SUCCESS,
        )

    def _extract_tool_calls(self, response: BaseResponse) -> List[ToolCall]:
        """Extract tool calls from a response.

        Args:
            response: Response from provider.

        Returns:
            List of ToolCall objects.
        """
        tool_calls = []

        # Check if response has choices with messages containing tool calls
        if response.choices:
            for choice in response.choices:
                message = choice.message

                # Check for tool_calls in message
                if message.tool_calls:
                    for tc in message.tool_calls:
                        # Handle different formats
                        if isinstance(tc, dict):
                            # Extract from dict format
                            if "function" in tc:
                                # OpenAI format
                                func = tc["function"]
                                arguments_str = func.get("arguments", "{}")
                                try:
                                    arguments = json.loads(arguments_str)
                                except json.JSONDecodeError:
                                    arguments = {}

                                tool_call = ToolCall(
                                    id=tc.get("id", ""),
                                    name=func.get("name", ""),
                                    arguments=arguments,
                                    raw_arguments=arguments_str,
                                )
                                tool_calls.append(tool_call)
                            else:
                                # Direct format
                                tool_call = ToolCall(
                                    id=tc.get("id", ""),
                                    name=tc.get("name", ""),
                                    arguments=tc.get("arguments", {}),
                                    raw_arguments=tc.get("raw_arguments"),
                                )
                                tool_calls.append(tool_call)

        return tool_calls

    def _create_assistant_message(
        self, response: BaseResponse, tool_calls: List[ToolCall]
    ) -> Dict[str, Any]:
        """Create assistant message with tool calls.

        Args:
            response: Original response from provider.
            tool_calls: List of tool calls to include.

        Returns:
            Message dict for conversation history.
        """
        # Format tool calls for message
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

        return {
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": tool_calls_data,
        }

    def format_tool_result_message(self, result: ToolResult) -> Dict[str, Any]:
        """Format tool result as a message.

        Args:
            result: Tool execution result.

        Returns:
            Message dict with tool result.
        """
        return {
            "role": "tool",
            "content": result.to_message_content(),
            "tool_call_id": result.tool_call_id,
        }

    def execute_all(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute all tool calls sequentially.

        Args:
            tool_calls: List of tool calls to execute.

        Returns:
            List of ToolResult objects.
        """
        results = []
        for tool_call in tool_calls:
            result = self.execute_tool_call(tool_call)
            results.append(result)
        return results

    def create_execution_entry(
        self,
        iteration: int,
        tool_call: ToolCall,
        tool_result: ToolResult,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolExecutionEntry:
        """Create an execution history entry.

        Args:
            iteration: The iteration number.
            tool_call: The tool call that was executed.
            tool_result: The result of execution.
            messages: Optional messages generated.

        Returns:
            ToolExecutionEntry for history tracking.
        """
        return ToolExecutionEntry(
            iteration=iteration,
            tool_call=tool_call,
            tool_result=tool_result,
            messages=messages or [],
        )

    @staticmethod
    def calculate_total_cost(execution_history: List[ToolExecutionEntry]) -> float:
        """Calculate total cost from execution history.

        Args:
            execution_history: List of tool execution entries.

        Returns:
            Total estimated cost in USD.
        """
        total_cost = 0.0
        for entry in execution_history:
            if entry.tool_result.cost is not None:
                total_cost += entry.tool_result.cost
        return total_cost
