import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, Iterator, List, Optional

import httpx

from justllms.core.models import Choice, Message, Role, Usage

if TYPE_CHECKING:
    from justllms.core.base import BaseProvider
    from justllms.core.completion import CompletionResponse


def parse_sse_stream(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    parse_chunk_fn: Callable[[str], Optional["StreamChunk"]],
    timeout: Optional[float] = None,
    error_prefix: str = "Streaming request",
) -> Iterator["StreamChunk"]:
    """Parse Server-Sent Events (SSE) stream from an HTTP endpoint.

    This is a shared helper for streaming responses that follow the SSE protocol.
    Both OpenAI-compatible and Azure OpenAI providers use this format.

    Args:
        url: API endpoint URL.
        payload: Request payload (should have stream=True).
        headers: Request headers including authorization.
        parse_chunk_fn: Callback to parse SSE line into StreamChunk.
        timeout: Optional timeout in seconds.
        error_prefix: Prefix for error messages (e.g., "OpenAI streaming request").

    Yields:
        StreamChunk objects parsed from the SSE stream.

    Raises:
        ProviderError: If the streaming request fails.
    """
    from justllms.exceptions import ProviderError

    try:
        with httpx.Client(timeout=timeout) as client, client.stream(
            "POST", url, json=payload, headers=headers
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                chunk = parse_chunk_fn(line)
                if chunk is not None:
                    yield chunk
                elif line.strip() == "data: [DONE]":
                    break
    except (httpx.HTTPError, httpx.RequestError) as e:
        raise ProviderError(f"{error_prefix} failed: {str(e)}") from e


class StreamChunk:
    """Individual chunk from a streaming response."""

    def __init__(
        self,
        content: Optional[str] = None,
        finish_reason: Optional[str] = None,
        usage: Optional[Usage] = None,
        raw: Any = None,
    ):
        """Initialize a stream chunk.

        Args:
            content: Text content in this chunk.
            finish_reason: Reason streaming stopped (if final chunk).
            usage: Token usage info (if available).
            raw: Raw provider response for debugging.
        """
        self.content = content
        self.finish_reason = finish_reason
        self.usage = usage
        self.raw = raw


class StreamResponse:
    """Accumulates streaming chunks and builds final CompletionResponse."""

    def __init__(self, provider: "BaseProvider", model: str, messages: List[Message]):
        """Initialize stream accumulator.

        Args:
            provider: Provider instance for cost estimation.
            model: Model name for token counting.
            messages: Original messages for token counting.
        """
        self.provider = provider
        self.model = model
        self.messages = messages
        self.id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        self.created = int(datetime.now().timestamp())
        self._content_chunks: List[str] = []
        self._finish_reason: Optional[str] = None
        self._usage: Optional[Usage] = None
        self.completed = False

    def accumulate(self, chunk: StreamChunk) -> None:
        """Accumulate content from a stream chunk.

        Args:
            chunk: Stream chunk to accumulate.
        """
        if chunk.content:
            self._content_chunks.append(chunk.content)
        if chunk.finish_reason:
            self._finish_reason = chunk.finish_reason
        if chunk.usage:
            self._usage = chunk.usage

    def mark_complete(self) -> None:
        """Mark stream as fully consumed."""
        self.completed = True

    def to_completion_response(self) -> "CompletionResponse":
        """Build CompletionResponse from accumulated chunks.

        Returns:
            CompletionResponse with proper structure.

        Raises:
            RuntimeError: If stream not fully consumed.
        """
        if not self.completed:
            raise RuntimeError(
                "Stream not fully consumed. Iterate through all chunks first "
                "or call drain() to consume remaining chunks."
            )

        from justllms.core.completion import CompletionResponse

        # Build Message
        message = Message(role=Role.ASSISTANT, content="".join(self._content_chunks))

        # Build Choice
        choice = Choice(index=0, message=message, finish_reason=self._finish_reason or "stop")

        # Build or estimate Usage
        if not self._usage:
            prompt_tokens = self.provider.count_message_tokens(self.messages, self.model)
            completion_tokens = self.provider.count_tokens(message.content, self.model)
            self._usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        # Set cost on Usage object
        self._usage.estimated_cost = self.provider.estimate_cost(self._usage, self.model)

        return CompletionResponse(
            id=self.id,
            model=self.model,
            choices=[choice],
            usage=self._usage,
            created=self.created,
            provider=self.provider.name,
        )


class SyncStreamResponse:
    """Synchronous streaming response."""

    def __init__(
        self,
        provider: "BaseProvider",
        model: str,
        messages: List[Message],
        raw_stream: Iterator[StreamChunk],
    ):
        """Initialize sync stream response.

        Args:
            provider: Provider instance.
            model: Model name.
            messages: Original messages.
            raw_stream: Iterator of StreamChunks.
        """
        self.accumulator = StreamResponse(provider, model, messages)
        self.raw_stream = raw_stream
        self._iterator_started = False

    def __iter__(self) -> Iterator[StreamChunk]:
        """Iterate over stream chunks.

        Yields:
            StreamChunk objects.

        Raises:
            RuntimeError: If iteration already started from a different iterator.
        """
        if self._iterator_started:
            raise RuntimeError(
                "Stream iteration already started. Cannot create multiple iterators. "
                "Use a single for loop or call get_final_response() to consume remaining chunks."
            )
        self._iterator_started = True

        for chunk in self.raw_stream:
            self.accumulator.accumulate(chunk)
            yield chunk

        self.accumulator.mark_complete()

    def drain(self) -> None:
        """Consume remaining chunks without yielding.

        Safe to call at any time - will consume from current position.
        """
        if self.accumulator.completed:
            return  # Already finished

        if not self._iterator_started:
            # Haven't started yet - consume entire stream
            for _ in self:
                pass
        else:
            # Already started - continue from current position
            for chunk in self.raw_stream:
                self.accumulator.accumulate(chunk)
            self.accumulator.mark_complete()

    def get_final_response(self) -> "CompletionResponse":
        """Get final CompletionResponse.

        Automatically drains stream if not yet fully consumed.

        Returns:
            CompletionResponse with cost and usage.
        """
        if not self.accumulator.completed:
            self.drain()
        return self.accumulator.to_completion_response()


class AsyncStreamResponse:
    """Asynchronous streaming response."""

    def __init__(
        self,
        provider: "BaseProvider",
        model: str,
        messages: List[Message],
        async_stream: AsyncIterator[StreamChunk],
    ):
        """Initialize async stream response.

        Args:
            provider: Provider instance.
            model: Model name.
            messages: Original messages.
            async_stream: Async iterator of StreamChunks.
        """
        self.accumulator = StreamResponse(provider, model, messages)
        self.async_stream = async_stream
        self._iterator_started = False

    async def __aiter__(self) -> AsyncIterator[StreamChunk]:
        """Async iterate over stream chunks.

        Yields:
            StreamChunk objects.

        Raises:
            RuntimeError: If iteration already started from a different iterator.
        """
        if self._iterator_started:
            raise RuntimeError(
                "Stream iteration already started. Cannot create multiple iterators. "
                "Use a single async for loop or call get_final_response() to consume remaining chunks."
            )
        self._iterator_started = True

        async for chunk in self.async_stream:
            self.accumulator.accumulate(chunk)
            yield chunk

        self.accumulator.mark_complete()

    async def drain(self) -> None:
        """Consume remaining chunks without yielding.

        Safe to call at any time - will consume from current position.
        """
        if self.accumulator.completed:
            return  # Already finished

        if not self._iterator_started:
            # Haven't started yet - consume entire stream
            async for _ in self:
                pass
        else:
            # Already started - continue from current position
            async for chunk in self.async_stream:
                self.accumulator.accumulate(chunk)
            self.accumulator.mark_complete()

    async def get_final_response(self) -> "CompletionResponse":
        """Get final CompletionResponse.

        Automatically drains stream if not yet fully consumed.

        Returns:
            CompletionResponse with cost and usage.
        """
        if not self.accumulator.completed:
            await self.drain()
        return self.accumulator.to_completion_response()
