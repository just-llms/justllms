"""Basic usage example for JustLLMs."""

import asyncio
from justllms import Client, Message

# Example 1: Simple completion
def simple_completion():
    """Basic completion example."""
    # Initialize client (will use environment variables for API keys)
    client = Client()
    
    # Create a completion
    response = client.completion.create(
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ],
        model="gpt-3.5-turbo",  # Will automatically route to OpenAI
    )
    
    print(f"Response: {response.content}")
    print(f"Provider: {response.provider}")
    print(f"Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")
    print(f"Cost: ${response.usage.estimated_cost:.4f}" if response.usage and response.usage.estimated_cost else "N/A")


# Example 2: Using different models
def multi_provider_example():
    """Example using multiple providers."""
    client = Client()
    
    # List of queries for different models
    queries = [
        ("openai/gpt-4o-mini", "Explain quantum computing in simple terms"),
        ("anthropic/claude-3-5-haiku-20241022", "Write a haiku about programming"),
        ("google/gemini-1.5-flash", "Summarize the theory of relativity"),
        ("gpt-3.5-turbo", "What is 2+2?"),  # Will route to cheapest provider
    ]
    
    for model, question in queries:
        response = client.completion.create(
            messages=[{"role": "user", "content": question}],
            model=model,
        )
        
        print(f"\nModel: {model}")
        print(f"Question: {question}")
        print(f"Response: {response.content}")
        print(f"Actual provider: {response.provider}/{response.model}")
        print("-" * 50)


# Example 3: Streaming responses
def streaming_example():
    """Example with streaming responses."""
    client = Client()
    
    print("Streaming response:")
    stream = client.completion.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a short story about a robot learning to paint."}
        ],
        model="gpt-3.5-turbo",
        stream=True,
        max_tokens=200,
    )
    
    for chunk in stream:
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")


# Example 4: Async usage
async def async_example():
    """Example using async methods."""
    client = Client()
    
    # Create multiple completions concurrently
    tasks = [
        client.completion.acreate(
            messages=[{"role": "user", "content": f"What is {i} + {i}?"}],
            model="gpt-3.5-turbo",
        )
        for i in range(1, 4)
    ]
    
    responses = await asyncio.gather(*tasks)
    
    for i, response in enumerate(responses, 1):
        print(f"Question {i}: {response.content}")


# Example 5: Using routing strategies
def routing_strategies_example():
    """Example with different routing strategies."""
    from justllms.routing import CostOptimizedStrategy, QualityOptimizedStrategy
    
    # Cost-optimized client
    cost_client = Client(
        config={
            "routing": {
                "strategy": "cost",
                "strategy_configs": {
                    "cost": {
                        "max_cost_per_1k_tokens": 0.001,
                    }
                }
            }
        }
    )
    
    # Quality-optimized client
    quality_client = Client(
        config={
            "routing": {
                "strategy": "quality",
                "strategy_configs": {
                    "quality": {
                        "min_quality_tier": "advanced",
                    }
                }
            }
        }
    )
    
    question = "Explain the theory of relativity"
    
    # Cost-optimized response
    cost_response = cost_client.completion.create(
        messages=[{"role": "user", "content": question}],
    )
    
    # Quality-optimized response
    quality_response = quality_client.completion.create(
        messages=[{"role": "user", "content": question}],
    )
    
    print(f"Cost-optimized model: {cost_response.provider}/{cost_response.model}")
    print(f"Quality-optimized model: {quality_response.provider}/{quality_response.model}")


# Example 6: Monitoring and cost tracking
def monitoring_example():
    """Example with monitoring and cost tracking."""
    client = Client(
        config={
            "monitoring": {
                "logging": {
                    "level": "DEBUG",
                    "console_output": True,
                },
                "cost_tracking": {
                    "budget_daily": 1.0,
                    "budget_per_request": 0.05,
                }
            }
        }
    )
    
    # Make some requests
    for i in range(3):
        response = client.completion.create(
            messages=[{"role": "user", "content": f"Count to {i+1}"}],
            model="gpt-3.5-turbo",
        )
    
    # Get monitoring summary
    summary = client.monitor.get_metrics_summary()
    print("\nMonitoring Summary:")
    print(f"Total requests: {sum(summary['metrics']['request_counts'].values())}")
    print(f"Cache hit rate: {summary['metrics']['cache_stats']['hit_rate']:.2%}")
    print(f"Daily cost: ${summary['cost_summary']['daily']['total_cost']:.4f}")


if __name__ == "__main__":
    print("=== JustLLMs Basic Usage Examples ===\n")
    
    print("1. Simple Completion:")
    simple_completion()
    
    print("\n2. Multi-Provider Example:")
    multi_provider_example()
    
    print("\n3. Streaming Example:")
    streaming_example()
    
    print("\n4. Async Example:")
    asyncio.run(async_example())
    
    print("\n5. Routing Strategies:")
    routing_strategies_example()
    
    print("\n6. Monitoring Example:")
    monitoring_example()