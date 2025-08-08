"""Advanced usage examples for JustLLMs."""

import asyncio
from pathlib import Path
from typing import List

from justllms import Client, Message
from justllms.routing import TaskBasedStrategy
from justllms.cache import DiskCacheBackend
from justllms.config import Config


# Example 1: Custom provider implementation
def custom_provider_example():
    """Example of adding a custom provider."""
    from justllms.core.base import BaseProvider, BaseResponse
    from justllms.core.models import Choice, Usage
    from justllms.providers import register_provider
    
    class CustomProvider(BaseProvider):
        """Custom provider implementation."""
        
        @property
        def name(self) -> str:
            return "custom"
        
        def get_available_models(self):
            return {
                "custom-model": {
                    "name": "custom-model",
                    "provider": "custom",
                    "max_tokens": 4096,
                }
            }
        
        def complete(self, messages, model, **kwargs):
            # Implement your custom logic here
            response_text = f"Custom response for: {messages[-1].content}"
            
            message = Message(role="assistant", content=response_text)
            choice = Choice(index=0, message=message, finish_reason="stop")
            usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            
            return BaseResponse(
                id="custom-123",
                model=model,
                choices=[choice],
                usage=usage,
            )
        
        async def acomplete(self, messages, model, **kwargs):
            # Async implementation
            return self.complete(messages, model, **kwargs)
        
        def stream(self, messages, model, **kwargs):
            # Streaming implementation
            yield self.complete(messages, model, **kwargs)
        
        async def astream(self, messages, model, **kwargs):
            # Async streaming implementation
            yield self.complete(messages, model, **kwargs)
    
    # Register the custom provider
    register_provider("custom", CustomProvider)
    
    # Use it
    client = Client()
    client.add_provider("custom", CustomProvider({"name": "custom"}))
    
    response = client.completion.create(
        messages=[{"role": "user", "content": "Hello custom provider!"}],
        model="custom/custom-model",
    )
    
    print(f"Custom provider response: {response.content}")


# Example 2: Custom routing strategy
def custom_routing_example():
    """Example of implementing a custom routing strategy."""
    from justllms.routing.strategies import RoutingStrategy
    
    class PriorityRoutingStrategy(RoutingStrategy):
        """Route based on predefined provider priorities."""
        
        def __init__(self, priorities: List[tuple[str, str]]):
            self.priorities = priorities
        
        def select(self, messages, providers, constraints=None, **kwargs):
            # Try providers in priority order
            for provider_name, model_name in self.priorities:
                if provider_name in providers:
                    provider = providers[provider_name]
                    if provider.validate_model(model_name):
                        return provider_name, model_name
            
            # Fallback to first available
            for name, provider in providers.items():
                models = provider.get_available_models()
                if models:
                    return name, list(models.keys())[0]
            
            raise ValueError("No suitable models found")
    
    # Use custom strategy
    client = Client()
    priority_strategy = PriorityRoutingStrategy([
        ("anthropic", "claude-3-5-sonnet-20241022"),
        ("openai", "gpt-4o"),
        ("openai", "gpt-3.5-turbo"),
    ])
    
    client.router.set_strategy(priority_strategy)
    
    response = client.completion.create(
        messages=[{"role": "user", "content": "Test priority routing"}],
    )
    
    print(f"Selected model: {response.provider}/{response.model}")


# Example 3: Advanced caching with semantic similarity
def semantic_caching_example():
    """Example of semantic caching (simplified)."""
    from justllms.cache.backends import BaseCacheBackend
    import hashlib
    
    class SemanticCacheBackend(BaseCacheBackend):
        """Simplified semantic cache using embeddings."""
        
        def __init__(self, similarity_threshold=0.95):
            self.cache = {}
            self.embeddings = {}  # In practice, use actual embeddings
            self.similarity_threshold = similarity_threshold
        
        def _get_embedding(self, text):
            # Simplified: use hash as "embedding"
            # In practice, use actual embedding model
            return hashlib.md5(text.encode()).hexdigest()
        
        def _calculate_similarity(self, emb1, emb2):
            # Simplified similarity calculation
            # In practice, use cosine similarity
            return 1.0 if emb1 == emb2 else 0.0
        
        def get(self, key):
            # First try exact match
            if key in self.cache:
                return self.cache[key]
            
            # Then try semantic similarity
            key_embedding = self._get_embedding(key)
            
            for cached_key, cached_value in self.cache.items():
                cached_embedding = self.embeddings.get(cached_key)
                if cached_embedding:
                    similarity = self._calculate_similarity(key_embedding, cached_embedding)
                    if similarity >= self.similarity_threshold:
                        return cached_value
            
            return None
        
        def set(self, key, value, ttl=None):
            self.cache[key] = value
            self.embeddings[key] = self._get_embedding(key)
        
        def delete(self, key):
            self.cache.pop(key, None)
            self.embeddings.pop(key, None)
        
        def clear(self):
            self.cache.clear()
            self.embeddings.clear()
        
        def exists(self, key):
            return key in self.cache
    
    # Use semantic cache
    semantic_cache = SemanticCacheBackend()
    client = Client(cache_manager=semantic_cache)
    
    # These similar queries might hit the cache
    queries = [
        "What is the capital of France?",
        "What's the capital city of France?",
        "Tell me the capital of France",
    ]
    
    for query in queries:
        response = client.completion.create(
            messages=[{"role": "user", "content": query}],
            model="gpt-3.5-turbo",
        )
        print(f"Query: {query}")
        print(f"Cached: {response.cached}")
        print(f"Response: {response.content}\n")


# Example 4: Budget-aware routing
def budget_aware_routing():
    """Example of budget-aware routing that switches models based on remaining budget."""
    import datetime
    
    class BudgetAwareClient(Client):
        """Client that automatically switches to cheaper models when budget is low."""
        
        def _create_completion(self, messages, model=None, **kwargs):
            # Check remaining budget
            daily_summary = self.monitor.cost_tracker.get_cost_summary("daily")
            daily_spent = daily_summary["total_cost"]
            daily_budget = self.monitor.cost_tracker.budget_daily or float('inf')
            
            remaining_budget = daily_budget - daily_spent
            budget_percentage = remaining_budget / daily_budget if daily_budget else 1.0
            
            # Switch to cheaper model if budget is low
            if budget_percentage < 0.2:  # Less than 20% budget remaining
                print(f"Low budget warning: {budget_percentage:.1%} remaining")
                if not model or "gpt-4" in model or "claude-3-opus" in model:
                    model = "gpt-3.5-turbo"  # Force cheaper model
                    print(f"Switching to cheaper model: {model}")
            
            return super()._create_completion(messages, model, **kwargs)
    
    # Use budget-aware client
    client = BudgetAwareClient(
        config={
            "monitoring": {
                "cost_tracking": {
                    "budget_daily": 1.0,
                }
            }
        }
    )
    
    # This would switch to cheaper models as budget depletes
    response = client.completion.create(
        messages=[{"role": "user", "content": "Explain machine learning"}],
        model="gpt-4o",  # Might be overridden if budget is low
    )
    
    print(f"Used model: {response.model}")


# Example 5: Multi-turn conversation with context management
def conversation_example():
    """Example of managing multi-turn conversations."""
    
    class ConversationManager:
        """Manage conversation context and token limits."""
        
        def __init__(self, client: Client, max_context_tokens: int = 4000):
            self.client = client
            self.max_context_tokens = max_context_tokens
            self.messages: List[Message] = []
        
        def add_message(self, role: str, content: str):
            """Add a message to the conversation."""
            self.messages.append(Message(role=role, content=content))
            self._trim_context()
        
        def _trim_context(self):
            """Trim conversation history to fit token limit."""
            from justllms.utils import count_tokens
            
            # Always keep system message if present
            system_messages = [m for m in self.messages if m.role == "system"]
            other_messages = [m for m in self.messages if m.role != "system"]
            
            # Trim from the beginning (keep recent context)
            while len(other_messages) > 1:
                total_tokens = count_tokens(system_messages + other_messages)
                if total_tokens <= self.max_context_tokens:
                    break
                other_messages.pop(0)
            
            self.messages = system_messages + other_messages
        
        def get_response(self, user_input: str, **kwargs):
            """Get a response for user input."""
            self.add_message("user", user_input)
            
            response = self.client.completion.create(
                messages=self.messages,
                **kwargs
            )
            
            # Add assistant response to history
            if response.content:
                self.add_message("assistant", response.content)
            
            return response
    
    # Use conversation manager
    client = Client()
    conversation = ConversationManager(client)
    
    # Add system prompt
    conversation.add_message(
        "system",
        "You are a helpful AI assistant. Keep responses concise."
    )
    
    # Multi-turn conversation
    exchanges = [
        "What is machine learning?",
        "Can you give me an example?",
        "How does it differ from traditional programming?",
    ]
    
    for user_input in exchanges:
        print(f"User: {user_input}")
        response = conversation.get_response(user_input, model="gpt-3.5-turbo")
        print(f"Assistant: {response.content}\n")


# Example 6: Parallel processing with rate limiting
async def parallel_processing_example():
    """Example of parallel processing with rate limiting."""
    import asyncio
    from asyncio import Semaphore
    
    class RateLimitedClient(Client):
        """Client with built-in rate limiting."""
        
        def __init__(self, *args, max_concurrent_requests=5, **kwargs):
            super().__init__(*args, **kwargs)
            self._semaphore = Semaphore(max_concurrent_requests)
        
        async def _acreate_completion(self, *args, **kwargs):
            async with self._semaphore:
                return await super()._acreate_completion(*args, **kwargs)
    
    # Process multiple requests in parallel with rate limiting
    client = RateLimitedClient(max_concurrent_requests=3)
    
    # Create 10 tasks
    tasks = []
    for i in range(10):
        task = client.completion.acreate(
            messages=[{"role": "user", "content": f"Count to {i+1}"}],
            model="gpt-3.5-turbo",
        )
        tasks.append(task)
    
    # Process with rate limiting
    print("Processing 10 requests with max 3 concurrent...")
    responses = await asyncio.gather(*tasks)
    
    for i, response in enumerate(responses):
        print(f"Task {i+1}: {response.content}")


if __name__ == "__main__":
    print("=== JustLLMs Advanced Usage Examples ===\n")
    
    print("1. Custom Provider Example:")
    custom_provider_example()
    
    print("\n2. Custom Routing Strategy:")
    custom_routing_example()
    
    print("\n3. Semantic Caching Example:")
    semantic_caching_example()
    
    print("\n4. Budget-Aware Routing:")
    budget_aware_routing()
    
    print("\n5. Conversation Management:")
    conversation_example()
    
    print("\n6. Parallel Processing with Rate Limiting:")
    asyncio.run(parallel_processing_example())