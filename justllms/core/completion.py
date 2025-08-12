"""Unified completion interface."""

from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from justllms.core.base import BaseResponse
from justllms.core.models import Choice, Message, Role, Usage

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

    def retrieve_and_complete(
        self,
        query: str,
        collection: str,
        messages: Optional[Union[List[Dict[str, Any]], List[Message]]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        context_template: Optional[str] = None,
        include_metadata: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[CompletionResponse, Iterator[CompletionResponse]]:
        """Retrieve relevant context and create completion."""
        if not self.client.retrieval:
            raise ValueError(
                "Retrieval not configured. Please initialize client with retrieval_config."
            )

        # Perform retrieval
        retrieval_result = self.client.retrieval.search(
            query=query,
            collection=collection,
            k=k,
            similarity_threshold=similarity_threshold,
            **kwargs,
        )

        # Build context from retrieved documents
        context_parts = []
        for doc in retrieval_result.documents:
            if include_metadata and doc.metadata:
                metadata_str = ", ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
                context_parts.append(f"[{metadata_str}]\n{doc.content}")
            else:
                context_parts.append(doc.content)

        retrieved_context = "\n\n".join(context_parts)

        # Use provided template or default
        if context_template is None:
            context_template = "Context:\n{context}\n\nQuestion: {query}\n\nPlease answer based on the provided context."

        # Format the context message
        context_message = context_template.format(context=retrieved_context, query=query)

        # Prepare messages
        if messages is None:
            formatted_messages = [Message(role=Role.USER, content=context_message)]
        else:
            formatted_messages = self._format_messages(messages)
            # Insert context as the first user message or prepend to existing first message
            if formatted_messages and formatted_messages[0].role == "user":
                # Prepend context to first user message
                formatted_messages[0] = Message(
                    role=Role.USER, content=f"{context_message}\n\n{formatted_messages[0].content}"
                )
            else:
                # Insert context as first message
                formatted_messages.insert(0, Message(role=Role.USER, content=context_message))

        # Create completion with enhanced messages
        params = {
            "messages": formatted_messages,
            "model": model,
            "provider": provider,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        # Filter out None values, but keep model=None for routing
        params = {k: v for k, v in params.items() if v is not None or k == "model"}

        if stream:
            return self.client._stream_completion(**params)
        else:
            response = self.client._create_completion(**params)
            # Add retrieval metadata to response
            if hasattr(response, "raw_response"):
                response.raw_response["retrieval_metadata"] = {
                    "query": query,
                    "collection": collection,
                    "documents_retrieved": len(retrieval_result.documents),
                    "search_time_ms": retrieval_result.search_time_ms,
                    "total_results": retrieval_result.total_results,
                }
            return response

    async def aretrieve_and_complete(
        self,
        query: str,
        collection: str,
        messages: Optional[Union[List[Dict[str, Any]], List[Message]]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        context_template: Optional[str] = None,
        include_metadata: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponse]]:
        """Retrieve relevant context and create completion (async)."""
        if not self.client.retrieval:
            raise ValueError(
                "Retrieval not configured. Please initialize client with retrieval_config."
            )

        # Perform retrieval
        retrieval_result = await self.client.retrieval.asearch(
            query=query,
            collection=collection,
            k=k,
            similarity_threshold=similarity_threshold,
            **kwargs,
        )

        # Build context from retrieved documents
        context_parts = []
        for doc in retrieval_result.documents:
            if include_metadata and doc.metadata:
                metadata_str = ", ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
                context_parts.append(f"[{metadata_str}]\n{doc.content}")
            else:
                context_parts.append(doc.content)

        retrieved_context = "\n\n".join(context_parts)

        # Use provided template or default
        if context_template is None:
            context_template = "Context:\n{context}\n\nQuestion: {query}\n\nPlease answer based on the provided context."

        # Format the context message
        context_message = context_template.format(context=retrieved_context, query=query)

        # Prepare messages
        if messages is None:
            formatted_messages = [Message(role=Role.USER, content=context_message)]
        else:
            formatted_messages = self._format_messages(messages)
            # Insert context as the first user message or prepend to existing first message
            if formatted_messages and formatted_messages[0].role == "user":
                # Prepend context to first user message
                formatted_messages[0] = Message(
                    role=Role.USER, content=f"{context_message}\n\n{formatted_messages[0].content}"
                )
            else:
                # Insert context as first message
                formatted_messages.insert(0, Message(role=Role.USER, content=context_message))

        # Create completion with enhanced messages
        params = {
            "messages": formatted_messages,
            "model": model,
            "provider": provider,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        # Filter out None values, but keep model=None for routing
        params = {k: v for k, v in params.items() if v is not None or k == "model"}

        if stream:
            return self.client._astream_completion(**params)
        else:
            response = await self.client._acreate_completion(**params)
            # Add retrieval metadata to response
            if hasattr(response, "raw_response"):
                response.raw_response["retrieval_metadata"] = {
                    "query": query,
                    "collection": collection,
                    "documents_retrieved": len(retrieval_result.documents),
                    "search_time_ms": retrieval_result.search_time_ms,
                    "total_results": retrieval_result.total_results,
                }
            return response
