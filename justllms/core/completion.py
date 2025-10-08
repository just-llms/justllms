from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union, overload

from justllms.core.base import BaseResponse
from justllms.core.models import Choice, Message, Usage
from justllms.utils.validators import validate_messages

if TYPE_CHECKING:
    from justllms.core.client import Client
    from justllms.core.streaming import AsyncStreamResponse, SyncStreamResponse


class CompletionResponse(BaseResponse):
    """Standard completion response format."""

    def __init__(
        self,
        id: str,
        model: str,
        choices: List[Choice],
        usage: Optional[Usage] = None,
        created: Optional[int] = None,
        system_fingerprint: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            id=id,
            model=model,
            choices=choices,
            usage=usage,
            created=created,
            system_fingerprint=system_fingerprint,
            **kwargs,
        )
        self.provider = provider
        self.tool_execution_history: Optional[List[Any]] = None
        self.tools_used: Optional[List[str]] = None
        self.tool_execution_cost: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "id": self.id,
            "model": self.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content,
                    },
                    "finish_reason": choice.finish_reason,
                }
                for choice in self.choices
            ],
            "usage": (
                {
                    "prompt_tokens": self.usage.prompt_tokens,
                    "completion_tokens": self.usage.completion_tokens,
                    "total_tokens": self.usage.total_tokens,
                    "estimated_cost": self.usage.estimated_cost,
                }
                if self.usage
                else None
            ),
            "created": self.created,
            "system_fingerprint": self.system_fingerprint,
            "provider": self.provider,
        }

    @property
    def content(self) -> str:
        """Get the content of the first choice."""
        if self.choices and self.choices[0].message:
            content = self.choices[0].message.content
            return content if isinstance(content, str) else str(content)
        return ""


class Completion:
    """Simplified completion interface with automatic fallbacks."""

    def __init__(self, client: "Client"):
        self.client = client

    @overload
    def create(
        self,
        messages: Union[List[Dict[str, Any]], List[Message]],
        *,
        stream: Literal[False] = False,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        user: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> CompletionResponse: ...

    @overload
    def create(
        self,
        messages: Union[List[Dict[str, Any]], List[Message]],
        *,
        stream: Literal[True],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        user: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> "Union[SyncStreamResponse, AsyncStreamResponse]": ...

    def create(
        self,
        messages: Union[List[Dict[str, Any]], List[Message]],
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        user: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> "Union[CompletionResponse, SyncStreamResponse, AsyncStreamResponse]":
        """Create a completion with automatic fallbacks.

        Args:
            messages: List of messages in the conversation.
            model: Specific model to use (optional, uses fallback or first available if not provided).
            provider: Specific provider to use (optional).
            stream: If True, returns streaming response instead of CompletionResponse.

            Common generation parameters (normalized across providers):
                temperature: Sampling temperature (0.0-2.0). Controls randomness.
                top_p: Nucleus sampling threshold (0.0-1.0).
                top_k: Top-k sampling limit (integer). Note: OpenAI doesn't support this natively.
                max_tokens: Maximum tokens to generate.
                stop: Stop sequence(s) - string or list of strings.
                n: Number of completions to generate (OpenAI only).
                presence_penalty: Penalize new tokens based on presence (-2.0 to 2.0).
                frequency_penalty: Penalize new tokens based on frequency (-2.0 to 2.0).

            Provider-specific parameters:
                generation_config: Gemini-only configuration dict. Supports:
                    - candidateCount: Number of response variations (int)
                    - responseMimeType: Output format, e.g., "application/json"
                    - responseSchema: Structured output schema (dict)
                    - thinkingConfig: {"thinkingBudget": int} for Gemini 2.5 models

            Advanced features (for future use):
                tools: Tool/function definitions (not fully implemented in v1).
                tool_choice: Control which tool to use.
                response_format: Response format specification (OpenAI).
                seed: Random seed for deterministic outputs (OpenAI).
                user: End-user identifier.
                timeout: Request timeout in seconds. If None, no timeout is enforced.

        Returns:
            CompletionResponse: The model's response.

        Examples:
            # OpenAI with common parameters
            response = client.completion.create(
                messages=[{"role": "user", "content": "Hello"}],
                provider="openai",
                temperature=0.7,
                max_tokens=100,
                n=1
            )

            # Gemini with common + provider-specific parameters
            response = client.completion.create(
                messages=[{"role": "user", "content": "Hello"}],
                provider="google",
                temperature=0.7,
                top_k=40,
                max_tokens=1024,
                generation_config={
                    "thinkingConfig": {"thinkingBudget": 100},
                    "responseMimeType": "application/json"
                }
            )
        """
        # Validate messages
        formatted_messages = validate_messages(messages)

        params = {
            "messages": formatted_messages,
            "model": model,
            "provider": provider,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "stop": stop,
            "n": n,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "generation_config": generation_config,
            "tools": tools,
            "tool_choice": tool_choice,
            "response_format": response_format,
            "seed": seed,
            "user": user,
            "timeout": timeout,
            **kwargs,
        }

        # Filter out None values, but keep model=None for fallback selection and stream=False
        params = {k: v for k, v in params.items() if v is not None or k in ("model", "stream")}

        return self.client._create_completion(**params)

    def _create_with_tools(
        self,
        messages: List[Message],
        tools: List[Any],  # Can be Tool objects or dicts
        provider: str,
        model: str,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
        execute_tools: bool = True,
        max_iterations: int = 10,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Create completion with tool execution loop.

        This method handles the full tool calling cycle:
        1. Format tools for the provider
        2. Call LLM with tools
        3. Extract tool calls from response
        4. Execute tools (if execute_tools=True)
        5. Add results to messages
        6. Repeat until no more tool calls or max_iterations reached

        Args:
            messages: Conversation messages.
            tools: List of Tool objects or tool definitions.
            provider: Provider name to use.
            model: Model name to use.
            tool_choice: Tool selection strategy.
            execute_tools: Whether to automatically execute tools.
            max_iterations: Maximum number of execution rounds.
            timeout: Optional timeout for LLM requests.
            **kwargs: Additional provider-specific parameters.

        Returns:
            CompletionResponse with tool execution history.
        """
        from justllms.tools.executor import ToolExecutor
        from justllms.tools.models import Tool

        # Get provider instance
        provider_instance = self.client.providers.get(provider)
        if not provider_instance:
            raise ValueError(f"Provider '{provider}' not found")

        # Get tool adapter for provider
        adapter = provider_instance.get_tool_adapter()
        if not adapter:
            raise ValueError(f"Provider '{provider}' does not support tools")

        # Convert tools to Tool objects if needed
        # Separate user tools and native tools
        tool_objects = []
        native_tool_objects = []

        for tool in tools:
            # Check if it's a native tool (e.g., GoogleSearch, GoogleCodeExecution)
            if (
                hasattr(tool, "is_native_tool")
                and callable(tool.is_native_tool)
                and tool.is_native_tool()
            ):
                native_tool_objects.append(tool)
                continue

            # Regular tool processing
            if isinstance(tool, Tool):
                tool_objects.append(tool)
            elif isinstance(tool, dict):
                # Pre-formatted tool dicts are not supported
                # Tools must be Tool objects or @tool decorated functions
                raise ValueError(
                    f"Pre-formatted tool dicts are not supported. "
                    f"Please use Tool objects or @tool decorated functions. "
                    f"Got dict: {tool.get('name', 'unknown')}"
                )
            else:
                # Assume it's a callable with .tool attribute from decorator
                if hasattr(tool, "tool"):
                    tool_objects.append(tool.tool)
                else:
                    raise ValueError(f"Invalid tool type: {type(tool)}")

        # Format tools for provider API
        # Pass native tools separately if adapter supports it
        if native_tool_objects and hasattr(adapter, "format_tools_with_native"):
            # NOTE: For Google Gemini, mixing native tools (GoogleSearch, GoogleCodeExecution)
            # with user-defined functions is only supported in the Live API, not REST API.
            # If both are provided, warn the user.
            if tool_objects and native_tool_objects and provider == "google":
                import warnings

                warnings.warn(
                    "Mixing native tools (GoogleSearch, GoogleCodeExecution) with user-defined "
                    "functions is only supported in Gemini's Live API, not the REST API. "
                    "This request may fail. Use either native tools OR user functions, not both. "
                    "See: https://ai.google.dev/gemini-api/docs/live-tools",
                    UserWarning,
                    stacklevel=2,
                )

            formatted_tools = adapter.format_tools_with_native(tool_objects, native_tool_objects)
        else:
            formatted_tools = adapter.format_tools_for_api(tool_objects)

        formatted_tool_choice = adapter.format_tool_choice(tool_choice)

        # Initialize executor
        executor = ToolExecutor(
            tools=tool_objects,
            timeout=(
                self.client.config.routing.tool_timeout
                if hasattr(self.client, "config") and hasattr(self.client.config, "routing")
                else 30.0
            ),
        )

        # Track execution history
        execution_history: List[Any] = []
        conversation_messages = list(messages)  # Copy to avoid mutation

        # Execution loop
        for iteration in range(max_iterations):
            # Call LLM with tools
            response = provider_instance.complete_with_tools(
                messages=conversation_messages,
                tools=formatted_tools,
                model=model,
                tool_choice=formatted_tool_choice,
                timeout=timeout,
                **kwargs,
            )

            # Extract tool calls - reconstruct full response dict for adapter
            # Adapters need choices, which are extracted into response.choices
            full_response_dict = {}
            if hasattr(response, "raw_response"):
                full_response_dict = dict(response.raw_response)

            # Add choices back into the dict for adapters that need them
            if response.choices:
                full_response_dict["choices"] = [
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role.value,
                            "content": choice.message.content,
                            "tool_calls": choice.message.tool_calls,
                        },
                        "finish_reason": choice.finish_reason,
                    }
                    for choice in response.choices
                ]

            tool_calls = adapter.extract_tool_calls(full_response_dict)

            # If no tool calls, we're done
            if not tool_calls:
                # Create final response with history
                final_response = CompletionResponse(
                    id=response.id,
                    model=response.model,
                    choices=response.choices,
                    usage=response.usage,
                    created=response.created,
                    system_fingerprint=response.system_fingerprint,
                    provider=provider,
                )
                final_response.tool_execution_history = execution_history
                final_response.tools_used = (
                    list(set(entry.tool_call.name for entry in execution_history))
                    if execution_history
                    else []
                )
                final_response.tool_execution_cost = executor.calculate_total_cost(
                    execution_history
                )
                return final_response

            # Execute tools if requested
            if not execute_tools:
                # Return response with tool calls but no execution
                final_response = CompletionResponse(
                    id=response.id,
                    model=response.model,
                    choices=response.choices,
                    usage=response.usage,
                    created=response.created,
                    system_fingerprint=response.system_fingerprint,
                    provider=provider,
                )
                final_response.tool_execution_history = execution_history
                final_response.tool_execution_cost = executor.calculate_total_cost(
                    execution_history
                )
                return final_response

            # Add assistant message with tool calls
            assistant_msg = adapter.format_tool_calls_message(tool_calls)
            if assistant_msg:
                conversation_messages.append(assistant_msg)

            # Execute each tool call
            for tool_call in tool_calls:
                # Execute tool
                tool_result = executor.execute_tool_call(tool_call)

                # Format result as message
                result_msg = adapter.format_tool_result_message(tool_result, tool_call)
                conversation_messages.append(result_msg)

                # Track in history
                entry = executor.create_execution_entry(
                    iteration=iteration,
                    tool_call=tool_call,
                    tool_result=tool_result,
                    messages=[],  # Can add generated messages here if needed
                )
                execution_history.append(entry)

        # If we hit max iterations, return last response
        final_response = CompletionResponse(
            id=response.id if "response" in locals() else "unknown",
            model=model,
            choices=response.choices if "response" in locals() else [],
            usage=response.usage if "response" in locals() else None,
            provider=provider,
        )
        final_response.tool_execution_history = execution_history
        final_response.tools_used = (
            list(set(entry.tool_call.name for entry in execution_history))
            if execution_history
            else []
        )
        final_response.tool_execution_cost = executor.calculate_total_cost(execution_history)
        return final_response
