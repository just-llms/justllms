from typing import Any, Dict, Optional


class GoogleNativeTool:
    """Base class for Google native tools."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize a Google native tool.

        Args:
            config: Optional configuration for the tool.
        """
        self.config = config or {}
        self._is_native = True
        self._provider = "google"

    def is_native_tool(self) -> bool:
        """Check if this is a native tool."""
        return True

    def get_provider(self) -> str:
        """Get the provider this tool belongs to."""
        return "google"

    def to_api_format(self) -> Dict[str, Any]:
        """Convert to Gemini API format.

        Should be overridden by subclasses.
        """
        raise NotImplementedError


class GoogleSearch(GoogleNativeTool):
    """Google Search native tool for Gemini.

    Enables server-side web search capabilities.

    Example:
        from justllms.tools.google import GoogleSearch

        response = client.completion.create(
            messages=[{"role": "user", "content": "What are the latest AI developments?"}],
            tools=[GoogleSearch()],
            provider="google"
        )
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Google Search tool.

        Args:
            config: Optional configuration including:
                - dynamic_retrieval_config: Configuration for dynamic retrieval
                    - mode: "MODE_DYNAMIC" or "MODE_UNSPECIFIED"
                    - dynamic_threshold: Threshold for retrieval (0.0-1.0)
        """
        super().__init__(config)
        self.name = "google_search"

    def to_api_format(self) -> Dict[str, Any]:
        """Convert to Gemini API format.

        Returns:
            Dict in format: {"google_search": {}}
            or {"google_search": {"dynamic_retrieval_config": {...}}}
        """
        if self.config:
            return {"google_search": self.config}
        return {"google_search": {}}


class GoogleCodeExecution(GoogleNativeTool):
    """Google Code Execution native tool for Gemini.

    Enables server-side Python code execution in a sandbox.

    Example:
        from justllms.tools.google import GoogleCodeExecution

        response = client.completion.create(
            messages=[{"role": "user", "content": "Calculate fibonacci sequence up to 100"}],
            tools=[GoogleCodeExecution()],
            provider="google"
        )
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Code Execution tool.

        Args:
            config: Optional configuration (currently unused for code execution).
        """
        super().__init__(config)
        self.name = "code_execution"

    def to_api_format(self) -> Dict[str, Any]:
        """Convert to Gemini API format.

        Returns:
            Dict in format: {"code_execution": {}}
        """
        if self.config:
            return {"code_execution": self.config}
        return {"code_execution": {}}
