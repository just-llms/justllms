"""Unified completion interface."""

from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from justllms.core.base import BaseResponse
from justllms.core.models import Choice, Message, Usage

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
        cached: bool = False,
        blocked: bool = False,
        validation_result: Optional[Any] = None,
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
        self.cached = cached
        self.blocked = blocked
        self.validation_result = validation_result

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
            "cached": self.cached,
        }


class Completion:
    """Unified completion interface for all providers."""

    def __init__(self, client: "Client"):
        self.client = client

    def create(
        self,
        messages: Union[List[Dict[str, Any]], List[Message]],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        user: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[CompletionResponse, Iterator[CompletionResponse]]:
        """Create a completion."""
        formatted_messages = self._format_messages(messages)

        params = {
            "messages": formatted_messages,
            "model": model,
            "provider": provider,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop,
            "tools": tools,
            "tool_choice": tool_choice,
            "response_format": response_format,
            "seed": seed,
            "user": user,
            **kwargs,
        }

        # Filter out None values, but keep model=None for routing
        params = {k: v for k, v in params.items() if v is not None or k == "model"}

        if stream:
            return self.client._stream_completion(**params)
        else:
            return self.client._create_completion(**params)

    async def acreate(
        self,
        messages: Union[List[Dict[str, Any]], List[Message]],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        user: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponse]]:
        """Create an async completion."""
        formatted_messages = self._format_messages(messages)

        params = {
            "messages": formatted_messages,
            "model": model,
            "provider": provider,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop,
            "tools": tools,
            "tool_choice": tool_choice,
            "response_format": response_format,
            "seed": seed,
            "user": user,
            **kwargs,
        }

        # Filter out None values, but keep model=None for routing
        params = {k: v for k, v in params.items() if v is not None or k == "model"}

        if stream:
            return self.client._astream_completion(**params)
        else:
            return await self.client._acreate_completion(**params)

    def _format_messages(
        self, messages: Union[List[Dict[str, Any]], List[Message]]
    ) -> List[Message]:
        """Format messages to Message objects."""
        if not messages:
            raise ValueError("Messages list cannot be empty - at least one message is required")

        if isinstance(messages[0], Message):
            return messages  # type: ignore

        formatted = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted.append(Message(**msg))
            else:
                formatted.append(msg)

        return formatted
