"""Tests for CompletionResponse serialization."""

from justllms.core.completion import CompletionResponse
from justllms.core.models import Choice, Message, Role, Usage
from justllms.tools.models import ToolCall, ToolExecutionEntry, ToolResult, ToolResultStatus


def _make_response(**kwargs) -> CompletionResponse:
    response = CompletionResponse(
        id="resp-1",
        model="gpt-test",
        choices=[
            Choice(
                index=0,
                message=Message(
                    role=Role.ASSISTANT,
                    content="Done",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                        }
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, estimated_cost=0.01),
        provider="openai",
    )
    for key, value in kwargs.items():
        setattr(response, key, value)
    return response


def test_to_dict_includes_tool_calls_in_message() -> None:
    data = _make_response().to_dict()

    message = data["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert message["content"] == "Done"
    assert message["tool_calls"][0]["id"] == "call_1"


def test_to_dict_includes_tool_metadata_when_set() -> None:
    entry = ToolExecutionEntry(
        iteration=0,
        tool_call=ToolCall(id="call_1", name="get_weather", arguments={"city": "NYC"}),
        tool_result=ToolResult(
            tool_call_id="call_1",
            status=ToolResultStatus.SUCCESS,
            result={"temp": 72},
            execution_time_ms=12.5,
            cost=0.001,
        ),
    )
    response = _make_response(
        tools_used=["get_weather"],
        tool_execution_cost=0.001,
        tool_execution_history=[entry],
    )

    data = response.to_dict()

    assert data["tools_used"] == ["get_weather"]
    assert data["tool_execution_cost"] == 0.001
    assert data["tool_execution_history"][0]["tool_call"]["name"] == "get_weather"
    assert data["tool_execution_history"][0]["result"] == {"temp": 72}


def test_to_dict_omits_tool_fields_when_none() -> None:
    data = _make_response().to_dict()

    assert "tools_used" not in data
    assert "tool_execution_cost" not in data
    assert "tool_execution_history" not in data
