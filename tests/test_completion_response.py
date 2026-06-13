"""Tests for CompletionResponse serialization."""

from justllms.core.completion import CompletionResponse
from justllms.core.models import Choice, Message, Role, Usage
from justllms.tools.models import ToolCall, ToolExecutionEntry, ToolResult, ToolResultStatus


def test_to_dict_includes_tool_calls_and_tool_call_id():
    response = CompletionResponse(
        id="resp-1",
        model="gpt-4o",
        choices=[
            Choice(
                index=0,
                message=Message(
                    role=Role.ASSISTANT,
                    content="",
                    tool_calls=[
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                ),
                finish_reason="tool_calls",
            ),
            Choice(
                index=1,
                message=Message(
                    role=Role.TOOL,
                    content='{"temp": 72}',
                    tool_call_id="call_abc",
                ),
                finish_reason=None,
            ),
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        provider="openai",
    )

    data = response.to_dict()

    assert data["choices"][0]["message"]["tool_calls"][0]["id"] == "call_abc"
    assert data["choices"][1]["message"]["tool_call_id"] == "call_abc"
    assert data["choices"][1]["message"]["role"] == "tool"


def test_to_dict_includes_tool_execution_metadata():
    entry = ToolExecutionEntry(
        iteration=1,
        tool_call=ToolCall(id="call_1", name="search", arguments={"q": "test"}),
        tool_result=ToolResult(
            tool_call_id="call_1",
            result={"items": []},
            status=ToolResultStatus.SUCCESS,
            execution_time_ms=12.5,
            cost=0.01,
        ),
    )
    response = CompletionResponse(
        id="resp-2",
        model="gpt-4o",
        choices=[
            Choice(
                index=0,
                message=Message(role=Role.ASSISTANT, content="Done."),
                finish_reason="stop",
            )
        ],
        provider="openai",
    )
    response.tools_used = ["search"]
    response.tool_execution_cost = 0.01
    response.tool_execution_history = [entry]

    data = response.to_dict()

    assert data["tools_used"] == ["search"]
    assert data["tool_execution_cost"] == 0.01
    assert data["tool_execution_history"] == [entry.to_dict()]


def test_to_dict_omits_tool_fields_when_unset():
    response = CompletionResponse(
        id="resp-3",
        model="gpt-4o",
        choices=[
            Choice(
                index=0,
                message=Message(role=Role.ASSISTANT, content="Hello"),
                finish_reason="stop",
            )
        ],
    )

    data = response.to_dict()

    assert "tools_used" not in data
    assert "tool_execution_cost" not in data
    assert "tool_execution_history" not in data
