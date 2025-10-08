import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union, get_args, get_origin

from justllms.tools.models import ParameterInfo

if TYPE_CHECKING:
    from justllms.tools.models import Tool


def python_type_to_json_schema(python_type: type) -> Dict[str, Any]:
    """Convert Python type hints to JSON Schema types.

    Args:
        python_type: Python type or type hint.

    Returns:
        JSON Schema type definition.
    """
    # Handle None type
    if python_type is type(None) or python_type is None:
        return {"type": "null"}

    # Handle basic types
    if python_type is str:
        return {"type": "string"}
    elif python_type is int:
        return {"type": "integer"}
    elif python_type is float:
        return {"type": "number"}
    elif python_type is bool:
        return {"type": "boolean"}
    elif python_type is dict or python_type is Dict:
        return {"type": "object"}

    # Handle typing module types
    origin = get_origin(python_type)
    args = get_args(python_type)

    # Handle Optional[T] (Union[T, None])
    if origin is Union:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            # Optional[T]
            schema = python_type_to_json_schema(non_none_args[0])
            # Don't mark as required in the parent schema
            return schema
        else:
            # Union of multiple types
            return {"oneOf": [python_type_to_json_schema(arg) for arg in args]}

    # Handle List[T]
    elif origin in (list, List):
        items_type = args[0] if args else Any
        return {
            "type": "array",
            "items": python_type_to_json_schema(items_type),
        }

    # Handle Dict[K, V]
    elif origin in (dict, Dict):
        return {"type": "object"}

    # Default to string for unknown types
    return {"type": "string"}


def extract_function_schema(func: Callable) -> Dict[str, ParameterInfo]:
    """Extract parameter information from a function signature.

    Args:
        func: The function to introspect.

    Returns:
        Dictionary mapping parameter names to ParameterInfo objects.
    """
    sig = inspect.signature(func)
    parameters = {}

    for param_name, param in sig.parameters.items():
        # Skip self/cls parameters
        if param_name in ("self", "cls"):
            continue

        # Determine if parameter is required
        required = param.default is inspect.Parameter.empty

        # Get type hint
        param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any

        # Convert to JSON schema type
        schema_info = python_type_to_json_schema(param_type)

        # Handle Optional types - they're not required
        if get_origin(param_type) is Union:
            args = get_args(param_type)
            if type(None) in args:
                required = False

        param_info = ParameterInfo(
            name=param_name,
            type=schema_info.get("type", "string"),
            required=required,
            default=None if param.default is inspect.Parameter.empty else param.default,
        )

        # Add additional schema properties
        if "items" in schema_info:
            param_info.items = schema_info["items"]
        if "properties" in schema_info:
            param_info.properties = schema_info["properties"]

        parameters[param_name] = param_info

    return parameters


def extract_docstring_descriptions(func: Callable) -> Dict[str, str]:
    """Extract parameter descriptions from function docstring.

    Supports Google-style docstrings:
        Args:
            param_name: Description of parameter.

    Args:
        func: Function to extract docstring from.

    Returns:
        Dictionary mapping parameter names to descriptions.
    """
    descriptions: Dict[str, str] = {}
    docstring = inspect.getdoc(func)

    if not docstring:
        return descriptions

    lines = docstring.split("\n")
    in_args_section = False
    current_param = None
    current_desc_lines = []

    for line in lines:
        stripped = line.strip()

        # Check if we're entering Args section
        if stripped in ("Args:", "Arguments:", "Parameters:"):
            in_args_section = True
            continue

        # Check if we're leaving Args section
        if (
            in_args_section
            and stripped
            and not line.startswith((" ", "\t"))
            and stripped.endswith(":")
        ):
            in_args_section = False

        if in_args_section:
            # Check if this is a parameter definition (has : after the param name)
            if ":" in stripped and line.startswith((" " * 4, "\t")):
                # Save previous parameter if exists
                if current_param and current_desc_lines:
                    descriptions[current_param] = " ".join(current_desc_lines).strip()

                # Parse new parameter
                param_part, desc_part = stripped.split(":", 1)
                current_param = param_part.strip()

                # Handle type hints in docstring (param_name (type): description)
                if "(" in current_param and ")" in current_param:
                    current_param = current_param.split("(")[0].strip()

                current_desc_lines = [desc_part.strip()] if desc_part.strip() else []

            elif current_param and stripped:
                # Continuation of previous parameter description
                current_desc_lines.append(stripped)

    # Save last parameter
    if current_param and current_desc_lines:
        descriptions[current_param] = " ".join(current_desc_lines).strip()

    return descriptions


def validate_tool_arguments(tool: "Tool", arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and coerce arguments for a tool.

    Args:
        tool: The Tool instance.
        arguments: Arguments to validate.

    Returns:
        Validated and coerced arguments.

    Raises:
        ValueError: If required arguments are missing or invalid.
    """

    validated = {}

    # Check required parameters
    for param_name, param_info in tool.parameters.items():
        if param_info.required and param_name not in arguments:
            raise ValueError(f"Missing required parameter: {param_name}")

        if param_name in arguments:
            value = arguments[param_name]

            # Basic type coercion
            if param_info.type == "integer" and not isinstance(value, int):
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    raise ValueError(f"Parameter {param_name} must be an integer") from None

            elif param_info.type == "number" and not isinstance(value, (int, float)):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    raise ValueError(f"Parameter {param_name} must be a number") from None

            elif param_info.type == "boolean" and not isinstance(value, bool):
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes")
                else:
                    value = bool(value)

            elif param_info.type == "array" and not isinstance(value, list):
                raise ValueError(f"Parameter {param_name} must be an array")

            elif param_info.type == "object" and not isinstance(value, dict):
                raise ValueError(f"Parameter {param_name} must be an object")

            validated[param_name] = value
        elif param_info.default is not None:
            # Use default value
            validated[param_name] = param_info.default

    return validated
