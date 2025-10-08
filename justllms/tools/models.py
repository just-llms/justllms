import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ParameterInfo(BaseModel):
    """Information about a tool parameter."""

    name: str
    type: str  # JSON Schema type
    description: Optional[str] = None
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None
    items: Optional[Dict[str, Any]] = None  # For array types
    properties: Optional[Dict[str, Any]] = None  # For object types


class Tool(BaseModel):
    """Core tool representation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    namespace: Optional[str] = None
    description: str
    callable: Callable
    parameters: Dict[str, ParameterInfo] = Field(default_factory=dict)
    parameter_descriptions: Dict[str, str] = Field(default_factory=dict)
    return_type: Optional[Any] = None  # Can be type or typing generic
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_native: bool = False  # For provider-specific native tools

    @property
    def full_name(self) -> str:
        """Get fully qualified name including namespace."""
        if self.namespace:
            return f"{self.namespace}.{self.name}"
        return self.name

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert tool to JSON Schema format for providers."""
        required_params = [name for name, param in self.parameters.items() if param.required]

        properties = {}
        for param_name, param_info in self.parameters.items():
            prop: Dict[str, Any] = {"type": param_info.type}

            # Add description from parameter_descriptions or param_info
            desc = self.parameter_descriptions.get(param_name) or param_info.description
            if desc:
                prop["description"] = desc

            if param_info.enum:
                prop["enum"] = param_info.enum
            if param_info.items:
                prop["items"] = param_info.items
            if param_info.properties:
                prop["properties"] = param_info.properties

            properties[param_name] = prop

        schema = {
            "type": "object",
            "properties": properties,
        }

        if required_params:
            schema["required"] = required_params

        return schema


class ToolCall(BaseModel):
    """Represents a tool invocation request from the LLM."""

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")
    name: str
    namespace: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    raw_arguments: Optional[str] = None  # Original JSON string from LLM

    @property
    def full_name(self) -> str:
        """Get fully qualified name including namespace."""
        if self.namespace:
            return f"{self.namespace}.{self.name}"
        return self.name


class ToolResultStatus(str, Enum):
    """Status of tool execution."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


class ToolResult(BaseModel):
    """Tool execution result."""

    tool_call_id: str
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    status: ToolResultStatus = ToolResultStatus.SUCCESS
    cost: Optional[float] = None
    """Estimated cost of tool execution in USD (e.g., API call costs)."""

    @property
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ToolResultStatus.SUCCESS

    def to_message_content(self) -> str:
        """Convert result to string for message content."""
        if self.is_success:
            if isinstance(self.result, str):
                return self.result
            elif self.result is None:
                return "Tool executed successfully with no output."
            else:
                import json

                try:
                    return json.dumps(self.result, default=str)
                except (TypeError, ValueError):
                    return str(self.result)
        else:
            return f"Error: {self.error}"


@dataclass
class ToolExecutionEntry:
    """Single entry in the tool execution history."""

    iteration: int
    tool_call: ToolCall
    tool_result: ToolResult
    messages: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "iteration": self.iteration,
            "tool_call": {
                "id": self.tool_call.id,
                "name": self.tool_call.name,
                "namespace": self.tool_call.namespace,
                "arguments": self.tool_call.arguments,
                "status": self.tool_result.status.value,
                "error": self.tool_result.error,
            },
            "execution_time_ms": self.tool_result.execution_time_ms,
            "result": self.tool_result.result if self.tool_result.is_success else None,
            "cost": self.tool_result.cost,
        }
