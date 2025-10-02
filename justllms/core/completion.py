from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from justllms.core.base import BaseResponse
from justllms.core.models import Choice, Message, Usage
from justllms.utils.validators import validate_messages

if TYPE_CHECKING:
    from justllms.core.client import Client


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
    """Simplified completion interface focused on intelligent routing."""

    def __init__(self, client: "Client"):
        self.client = client

    def create(
        self,
        messages: Union[List[Dict[str, Any]], List[Message]],
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
    ) -> CompletionResponse:
        """Create a completion with intelligent routing.

        Args:
            messages: List of messages in the conversation.
            model: Specific model to use (optional, will be routed automatically if not provided).
            provider: Specific provider to use (optional).

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

        # Filter out None values, but keep model=None for routing
        params = {k: v for k, v in params.items() if v is not None or k == "model"}

        return self.client._create_completion(**params)
