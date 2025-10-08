import inspect
from typing import Callable, Dict, Optional, Union

from justllms.tools.models import Tool
from justllms.tools.utils import (
    extract_docstring_descriptions,
    extract_function_schema,
)


def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    namespace: Optional[str] = None,
    description: Optional[str] = None,
    parameter_descriptions: Optional[Dict[str, str]] = None,
    register: bool = False,
) -> Union[Callable, Tool]:
    """Decorator to convert a function into a Tool.

    Can be used with or without parentheses:
        @tool
        def my_func(): ...

        @tool(name="custom_name", namespace="math")
        def my_func(): ...

    Args:
        func: The function to convert (when used without parentheses).
        name: Custom name for the tool (defaults to function name).
        namespace: Optional namespace for the tool.
        description: Custom description (defaults to function docstring).
        parameter_descriptions: Additional parameter descriptions.
        register: Whether to register globally (default: False).

    Returns:
        Tool instance when used as decorator, or decorator function.

    Examples:
        >>> @tool
        ... def add(a: int, b: int) -> int:
        ...     '''Add two numbers.'''
        ...     return a + b

        >>> @tool(namespace="math", description="Multiply numbers")
        ... def multiply(x: float, y: float) -> float:
        ...     return x * y

        >>> @tool(
        ...     name="search_documents",
        ...     parameter_descriptions={
        ...         "query": "The search query",
        ...         "limit": "Maximum number of results"
        ...     }
        ... )
        ... def search(query: str, limit: int = 10) -> list:
        ...     return [f"Result for {query}"]
    """

    def decorator(f: Callable) -> Tool:
        """Inner decorator that creates the Tool instance."""
        # Extract metadata
        tool_name = name or f.__name__
        tool_description = description or inspect.getdoc(f) or f"Tool: {tool_name}"

        # Extract parameters
        parameters = extract_function_schema(f)

        # Extract parameter descriptions from docstring
        docstring_descriptions = extract_docstring_descriptions(f)

        # Merge parameter descriptions
        merged_descriptions = {}
        if docstring_descriptions:
            merged_descriptions.update(docstring_descriptions)
        if parameter_descriptions:
            merged_descriptions.update(parameter_descriptions)

        # Update parameter descriptions in ParameterInfo objects
        for param_name, param_info in parameters.items():
            if param_name in merged_descriptions:
                param_info.description = merged_descriptions[param_name]

        # Get return type
        sig = inspect.signature(f)
        return_type = (
            sig.return_annotation if sig.return_annotation != inspect.Signature.empty else None
        )

        # Create Tool instance
        tool_instance = Tool(
            name=tool_name,
            namespace=namespace,
            description=tool_description,
            callable=f,
            parameters=parameters,
            parameter_descriptions=merged_descriptions,
            return_type=return_type,
        )

        # Register globally if requested
        if register:
            from justllms.tools.registry import GlobalToolRegistry

            registry = GlobalToolRegistry()
            registry.register(tool_instance)

        # Add the tool as an attribute of the function
        f.tool = tool_instance  # type: ignore

        return tool_instance

    # Handle usage with or without parentheses
    if func is not None:
        # Used without parentheses: @tool
        return decorator(func)
    else:
        # Used with parentheses: @tool(...)
        return decorator


def tool_from_callable(
    func: Callable,
    name: Optional[str] = None,
    namespace: Optional[str] = None,
    description: Optional[str] = None,
    parameter_descriptions: Optional[Dict[str, str]] = None,
) -> Tool:
    """Convert an existing callable into a Tool.

    This is useful for converting existing functions that you can't
    or don't want to decorate directly.

    Args:
        func: The callable to convert.
        name: Custom name for the tool (defaults to function name).
        namespace: Optional namespace for the tool.
        description: Custom description (defaults to function docstring).
        parameter_descriptions: Parameter descriptions.

    Returns:
        Tool instance.

    Examples:
        >>> def existing_func(x: int, y: int) -> int:
        ...     return x * y
        ...
        >>> tool_instance = tool_from_callable(
        ...     existing_func,
        ...     name="multiplier",
        ...     description="Multiplies two numbers"
        ... )
    """
    # Extract metadata
    tool_name = name or func.__name__
    tool_description = description or inspect.getdoc(func) or f"Tool: {tool_name}"

    # Extract parameters
    parameters = extract_function_schema(func)

    # Extract parameter descriptions from docstring
    docstring_descriptions = extract_docstring_descriptions(func)

    # Merge parameter descriptions
    merged_descriptions = {}
    if docstring_descriptions:
        merged_descriptions.update(docstring_descriptions)
    if parameter_descriptions:
        merged_descriptions.update(parameter_descriptions)

    # Update parameter descriptions in ParameterInfo objects
    for param_name, param_info in parameters.items():
        if param_name in merged_descriptions:
            param_info.description = merged_descriptions[param_name]

    # Get return type
    sig = inspect.signature(func)
    return_type = (
        sig.return_annotation if sig.return_annotation != inspect.Signature.empty else None
    )

    # Create Tool instance
    return Tool(
        name=tool_name,
        namespace=namespace,
        description=tool_description,
        callable=func,
        parameters=parameters,
        parameter_descriptions=merged_descriptions,
        return_type=return_type,
    )
