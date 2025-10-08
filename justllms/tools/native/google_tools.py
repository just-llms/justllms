from typing import Any, Dict, Optional

from justllms.tools.models import Tool


class GoogleNativeTool(Tool):
    """Base class for Google's native tools.

    Native tools are executed server-side by Google's API and don't require
    local execution. They're enabled via provider configuration.
    """

    def __init__(
        self,
        name: str,
        description: str,
        namespace: str = "google",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a Google native tool.

        Args:
            name: Tool identifier.
            description: Human-readable description.
            namespace: Tool namespace (default: "google").
            config: Provider-specific configuration.
        """

        # Native tools don't have a callable - they're handled by the provider
        def _placeholder() -> None:
            raise NotImplementedError(
                f"Native tool '{name}' is executed server-side by Google. "
                "It should not be called directly."
            )

        # Store provider and config in metadata
        metadata = config.copy() if config else {}
        metadata["provider"] = "google"

        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            callable=_placeholder,
            parameters={},
            parameter_descriptions={},
            return_type=None,
            metadata=metadata,
            is_native=True,
        )


class GoogleSearch(GoogleNativeTool):
    """Google Search tool for Gemini models.

    Enables real-time web search capability. Results are retrieved and
    processed by Google's servers before being returned in the LLM response.

    Configuration options:
        - dynamic_retrieval_config: Configure retrieval parameters
          - mode: "MODE_DYNAMIC" (default) or "MODE_UNSPECIFIED"
          - dynamic_threshold: float (0.0-1.0), relevance threshold

    Example:
        ```python
        # Enable Google Search in config
        config = {
            "providers": {
                "google": {
                    "api_key": "...",
                    "native_tools": {
                        "google_search": {
                            "enabled": True,
                            "dynamic_retrieval_config": {
                                "mode": "MODE_DYNAMIC",
                                "dynamic_threshold": 0.7
                            }
                        }
                    }
                }
            }
        }

        client = Client(config)
        response = client.completion.create(
            messages=[{"role": "user", "content": "What's the weather in SF?"}],
            provider="google",
            model="gemini-2.0-flash-exp",
        )
        ```

    Note:
        - Only available in Gemini 1.5 Pro and newer models
        - Requires appropriate API permissions
        - Search results count toward token usage
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Google Search tool.

        Args:
            config: Optional configuration dict with:
                - dynamic_retrieval_config: Retrieval settings
                - enabled: Whether tool is enabled (default: True)
        """
        super().__init__(
            name="google_search",
            description=(
                "Search Google for real-time information. Returns relevant web results "
                "that are automatically incorporated into the response."
            ),
            namespace="google",
            config=config,
        )

    def to_api_format(self) -> Dict[str, Any]:
        """Convert to Gemini API format.

        Returns:
            Dict in format expected by Gemini API's googleSearch tool.
        """
        api_format: Dict[str, Any] = {"googleSearch": {}}

        # Add dynamic retrieval config if provided
        if "dynamic_retrieval_config" in self.metadata:
            api_format["googleSearchRetrieval"] = {
                "dynamicRetrievalConfig": self.metadata["dynamic_retrieval_config"]
            }

        return api_format


class GoogleCodeExecution(GoogleNativeTool):
    """Code execution tool for Gemini models.

    Enables Python code execution in a sandboxed server-side environment.
    The model can generate and execute code to solve problems, perform
    calculations, or process data.

    Configuration options:
        - timeout: Execution timeout in seconds (default: 30)
        - max_output_size: Maximum output size in bytes

    Example:
        ```python
        # Enable code execution in config
        config = {
            "providers": {
                "google": {
                    "api_key": "...",
                    "native_tools": {
                        "code_execution": {
                            "enabled": True,
                            "timeout": 30
                        }
                    }
                }
            }
        }

        client = Client(config)
        response = client.completion.create(
            messages=[{"role": "user", "content": "Calculate fibonacci(20)"}],
            provider="google",
            model="gemini-2.0-flash-exp",
        )
        ```

    Note:
        - Only available in Gemini 1.5 Pro and newer models
        - Execution happens in isolated sandbox environment
        - Code output is included in response
        - Limited to Python standard library
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize code execution tool.

        Args:
            config: Optional configuration dict with:
                - timeout: Execution timeout in seconds
                - max_output_size: Maximum output size
                - enabled: Whether tool is enabled (default: True)
        """
        super().__init__(
            name="code_execution",
            description=(
                "Execute Python code in a sandboxed environment. "
                "Use this to perform calculations, data processing, or algorithmic tasks."
            ),
            namespace="google",
            config=config,
        )

    def to_api_format(self) -> Dict[str, Any]:
        """Convert to Gemini API format.

        Returns:
            Dict in format expected by Gemini API's codeExecution tool.
        """
        return {"codeExecution": {}}


# Registry of available Google native tools
GOOGLE_NATIVE_TOOLS = {
    "google_search": GoogleSearch,
    "code_execution": GoogleCodeExecution,
}


def get_google_native_tool(name: str, config: Optional[Dict[str, Any]] = None) -> GoogleNativeTool:
    """Get a Google native tool by name.

    Args:
        name: Tool name ("google_search" or "code_execution").
        config: Optional tool-specific configuration.

    Returns:
        GoogleNativeTool instance.

    Raises:
        ValueError: If tool name is not recognized.
    """
    if name not in GOOGLE_NATIVE_TOOLS:
        raise ValueError(
            f"Unknown Google native tool: {name}. "
            f"Available tools: {list(GOOGLE_NATIVE_TOOLS.keys())}"
        )

    tool_class = GOOGLE_NATIVE_TOOLS[name]
    return tool_class(config=config)
