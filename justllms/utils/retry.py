"""Retry utilities for handling API failures."""

import asyncio
import random
import time
from typing import Any, Callable, Optional, TypeVar, Union

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
)

from justllms.exceptions import ProviderError, RateLimitError, TimeoutError

T = TypeVar("T")


class RetryHandler:
    """Configurable retry handler for API calls."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def should_retry(self, exception: Exception) -> bool:
        """Determine if an exception should trigger a retry."""
        # Always retry on rate limits and timeouts
        if isinstance(exception, (RateLimitError, TimeoutError)):
            return True

        # Retry on specific provider errors
        if isinstance(exception, ProviderError):
            error_str = str(exception).lower()
            retryable_errors = [
                "rate limit",
                "timeout",
                "connection",
                "server error",
                "502",
                "503",
                "504",
                "429",
            ]
            return any(error in error_str for error in retryable_errors)

        # Retry on connection errors
        return bool(isinstance(exception, (ConnectionError, asyncio.TimeoutError)))

    def get_retry_delay(self, attempt: int) -> float:
        """Calculate the retry delay for a given attempt."""
        delay = self.initial_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add random jitter (±25%)
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    def create_decorator(self) -> Any:
        """Create a tenacity retry decorator with this configuration."""
        if self.jitter:
            wait_strategy = wait_random_exponential(
                multiplier=self.initial_delay,
                max=self.max_delay,
            )
        else:
            wait_strategy = wait_exponential(  # type: ignore
                multiplier=self.initial_delay,
                max=self.max_delay,
            )

        return retry(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_strategy,
            retry=retry_if_exception_type(
                (
                    RateLimitError,
                    TimeoutError,
                    ConnectionError,
                    asyncio.TimeoutError,
                )
            ),
        )

    def sync_retry(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Synchronously retry a function."""
        last_exception = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if not self.should_retry(e) or attempt == self.max_attempts:
                    raise

                delay = self.get_retry_delay(attempt)
                time.sleep(delay)

        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Retry failed with no exception captured")

    async def async_retry(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Asynchronously retry a function."""
        last_exception = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)  # type: ignore
                else:
                    return func(*args, **kwargs)  # type: ignore
            except Exception as e:
                last_exception = e

                if not self.should_retry(e) or attempt == self.max_attempts:
                    raise

                delay = self.get_retry_delay(attempt)
                await asyncio.sleep(delay)

        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Retry failed with no exception captured")


def exponential_backoff(
    func: Optional[Callable] = None,
    *,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,),
) -> Union[Callable, Any]:
    """Decorator for exponential backoff retry logic."""

    def decorator(f: Callable) -> Callable:
        handler = RetryHandler(
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
        )

        if asyncio.iscoroutinefunction(f):

            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await handler.async_retry(f, *args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return handler.sync_retry(f, *args, **kwargs)

            return sync_wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)
