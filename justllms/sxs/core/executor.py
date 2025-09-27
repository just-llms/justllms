import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from justllms.sxs.models import ModelResponse, ResponseStatus

logger = logging.getLogger(__name__)


class ParallelExecutor:
    """Execute multiple model calls in parallel."""

    def __init__(self, client: Any) -> None:
        """Initialize the executor.

        Args:
            client: JustLLM client instance
        """
        self.client = client

    def execute_comparison(
        self,
        prompt: str,
        models: List[Tuple[str, str]],
        on_model_complete: Optional[Callable] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, ModelResponse]:
        """Execute all models in parallel.

        Args:
            prompt: The prompt to send to all models
            models: List of (provider, model) tuples
            on_model_complete: Callback when a model completes
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary mapping model_id to ModelResponse
        """
        results: Dict[str, ModelResponse] = {}

        if not models:
            return results

        def call_model(provider: str, model: str) -> Tuple[str, ModelResponse]:
            """Call a single model."""
            model_id = f"{provider}/{model}"
            start = time.time()

            try:
                # Call the model
                response = self.client.completion.create(
                    messages=[{"role": "user", "content": prompt}],
                    provider=provider,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Create successful response
                result = ModelResponse(
                    provider=provider,
                    model=model,
                    content=response.content,
                    status=ResponseStatus.COMPLETED,
                    latency=time.time() - start,
                    tokens=response.usage.total_tokens if response.usage else 0,
                    cost=response.usage.estimated_cost if response.usage else 0.0,
                )

            except Exception as e:
                # Create error response
                logger.error(f"Error calling {model_id}: {e}")
                result = ModelResponse(
                    provider=provider,
                    model=model,
                    content="",
                    status=ResponseStatus.ERROR,
                    latency=time.time() - start,
                    tokens=0,
                    cost=0.0,
                    error=str(e),
                )

            # Call callback if provided
            if on_model_complete:
                try:
                    on_model_complete(model_id, result)
                except Exception as e:
                    logger.error(f"Error in callback for {model_id}: {e}")

            return model_id, result

        with ThreadPoolExecutor(max_workers=min(len(models), 10)) as executor:
            futures = [executor.submit(call_model, provider, model) for provider, model in models]

            try:
                for future in as_completed(futures):
                    try:
                        model_id, result = future.result()
                        results[model_id] = result
                    except Exception as e:
                        logger.error(f"Error processing future: {e}")
            finally:
                for future in futures:
                    if not future.done():
                        future.cancel()

                for provider, model in models:
                    model_id = f"{provider}/{model}"
                    if model_id not in results:
                        results[model_id] = ModelResponse(
                            provider=provider,
                            model=model,
                            content="",
                            status=ResponseStatus.ERROR,
                            latency=0.0,
                            tokens=0,
                            cost=0.0,
                            error="Request failed or was cancelled",
                        )

        return results
