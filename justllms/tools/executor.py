import json
import logging
import multiprocessing as mp
import pickle
import threading
import time
from multiprocessing.process import BaseProcess
from typing import Any, Callable, Dict, List, Optional, Tuple

from justllms.core.base import BaseResponse
from justllms.tools.models import Tool, ToolCall, ToolExecutionEntry, ToolResult, ToolResultStatus
from justllms.tools.utils import validate_tool_arguments

logger = logging.getLogger(__name__)


def _run_tool_worker(
    result_queue: mp.Queue[Tuple[str, Any]],
    func: Callable[..., Any],
    kwargs: Dict[str, Any],
) -> None:
    """Run a tool callable in an isolated process and return the outcome."""
    try:
        result_queue.put(("success", func(**kwargs)))
    except Exception as exc:
        result_queue.put(("error", str(exc)))


class ToolExecutor:
    """Executes tools sequentially with error handling.

    This executor handles tool validation, argument parsing, execution,
    and error recovery. It does NOT support parallel execution - all
    tools run sequentially.

    Attributes:
        tools: Dictionary mapping tool names to Tool instances.
        timeout: Maximum execution time per tool in seconds.

    Note:
        Timeouts terminate picklable tools in a subprocess. Non-picklable
        callables fall back to a daemon thread where timeout only stops
        waiting and cannot guarantee cancellation.
    """

    def __init__(
        self,
        tools: List[Tool],
        timeout: float = 30.0,
    ):
        """Initialize the tool executor.

        Args:
            tools: List of Tool instances available for execution.
            timeout: Maximum execution time per tool in seconds.
        """
        self.tools = {tool.name: tool for tool in tools}
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
        result, error, timed_out = self._execute_callable_with_timeout(
            tool.callable, validated_args, tool_name=tool_call.name
        )
        execution_time_ms = (time.time() - start_time) * 1000

        if timed_out:
            return ToolResult(
                tool_call_id=tool_call.id,
                result=None,
                error=error,
                execution_time_ms=execution_time_ms,
                status=ToolResultStatus.TIMEOUT,
            )

        if error is not None:
            return ToolResult(
                tool_call_id=tool_call.id,
                result=None,
                error=error,
                execution_time_ms=execution_time_ms,
                status=ToolResultStatus.ERROR,
            )

        return ToolResult(
            tool_call_id=tool_call.id,
            result=result,
            error=None,
            execution_time_ms=execution_time_ms,
            status=ToolResultStatus.SUCCESS,
        )

    def _execute_callable_with_timeout(
        self,
        callable_fn: Callable[..., Any],
        validated_args: Dict[str, Any],
        tool_name: str,
    ) -> Tuple[Any, Optional[str], bool]:
        """Execute a callable with timeout enforcement.

        Returns:
            Tuple of (result, error_message, timed_out).
        """
        try:
            pickle.dumps(callable_fn)
        except (pickle.PicklingError, TypeError):
            logger.debug(
                "Tool '%s' is not picklable; using thread-based timeout fallback",
                tool_name,
            )
            return self._execute_in_thread(callable_fn, validated_args, tool_name)

        return self._execute_in_process(callable_fn, validated_args, tool_name)

    def _execute_in_process(
        self,
        callable_fn: Callable[..., Any],
        validated_args: Dict[str, Any],
        tool_name: str,
    ) -> Tuple[Any, Optional[str], bool]:
        """Execute a picklable callable in a subprocess that can be terminated."""
        ctx = mp.get_context("spawn")
        result_queue: mp.Queue[Tuple[str, Any]] = ctx.Queue()
        process = ctx.Process(
            target=_run_tool_worker,
            args=(result_queue, callable_fn, validated_args),
        )
        process.start()
        process.join(timeout=self.timeout)

        if process.is_alive():
            self._terminate_process(process)
            logger.warning(
                "Tool '%s' exceeded timeout of %ss and was terminated",
                tool_name,
                self.timeout,
            )
            return (
                None,
                f"Tool execution timed out after {self.timeout}s",
                True,
            )

        if not result_queue.empty():
            status, payload = result_queue.get_nowait()
            if status == "success":
                return payload, None, False
            return None, payload, False

        exit_code = process.exitcode
        return (
            None,
            f"Tool process exited without returning a result (exit code: {exit_code})",
            False,
        )

    def _execute_in_thread(
        self,
        callable_fn: Callable[..., Any],
        validated_args: Dict[str, Any],
        tool_name: str,
    ) -> Tuple[Any, Optional[str], bool]:
        """Best-effort timeout for callables that cannot run in a subprocess."""
        result_container: Dict[str, Any] = {}

        def execute_tool() -> None:
            try:
                result_container["result"] = callable_fn(**validated_args)
                result_container["success"] = True
            except Exception as exc:
                result_container["error"] = str(exc)
                result_container["success"] = False

        thread = threading.Thread(target=execute_tool, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            logger.warning(
                "Tool '%s' exceeded timeout of %ss; execution may continue in background "
                "because the callable is not picklable for process termination",
                tool_name,
                self.timeout,
            )
            return (
                None,
                (
                    f"Tool execution timed out after {self.timeout}s "
                    "(best-effort; execution may continue in background)"
                ),
                True,
            )

        if not result_container.get("success", False):
            return None, result_container.get("error", "Unknown error"), False

        return result_container.get("result"), None, False

    @staticmethod
    def _terminate_process(process: BaseProcess) -> None:
        """Terminate a subprocess, escalating to kill if needed."""
        if not process.is_alive():
            return

        process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join()

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
