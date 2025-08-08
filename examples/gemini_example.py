"""Examples of using Google Gemini models with JustLLMs."""

import asyncio
from justllms import Client

def basic_gemini_example():
    """Basic example using Gemini models."""
    client = Client()
    
    # Using Gemini 1.5 Pro (flagship model)
    response = client.completion.create(
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Explain the concept of machine learning in simple terms."}
        ],
        model="google/gemini-1.5-pro",
        temperature=0.7,
        max_tokens=500
    )
    
    print(f"Gemini 1.5 Pro Response:\n{response.content}\n")
    print(f"Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")
    print(f"Cost: ${response.usage.estimated_cost:.6f}" if response.usage and response.usage.estimated_cost else "N/A")


def gemini_model_comparison():
    """Compare different Gemini models for the same task."""
    client = Client()
    
    models = [
        "google/gemini-2.5-flash",     # Latest with 65k output tokens
        "google/gemini-1.5-pro",       # Most capable, higher cost
        "google/gemini-1.5-flash",     # Balanced performance and cost
        "google/gemini-1.5-flash-8b",  # Fastest and cheapest
        "google/gemini-1.0-pro"        # Stable, general-purpose
    ]
    
    question = "Write a Python function to calculate factorial"
    
    for model in models:
        try:
            response = client.completion.create(
                messages=[{"role": "user", "content": question}],
                model=model,
                max_tokens=200
            )
            
            print(f"\n{'='*50}")
            print(f"Model: {model}")
            print(f"Response:\n{response.content[:200]}...")
            if response.usage:
                print(f"Tokens: {response.usage.total_tokens}")
                print(f"Cost: ${response.usage.estimated_cost:.6f}")
        except Exception as e:
            print(f"Error with {model}: {e}")


def gemini_long_context_example():
    """Demonstrate Gemini's long context capabilities."""
    client = Client()
    
    # Gemini 1.5 Pro supports up to 2M tokens context
    # Gemini 1.5 Flash supports up to 1M tokens context
    
    # Create a long document (simulated)
    long_document = """
    Chapter 1: Introduction to Neural Networks
    
    Neural networks are computational models inspired by the human brain...
    """ * 100  # Repeat to create a longer document
    
    response = client.completion.create(
        messages=[
            {"role": "user", "content": f"Please summarize this document:\n\n{long_document}"}
        ],
        model="google/gemini-1.5-flash",  # Good for long context at lower cost
        max_tokens=500
    )
    
    print("Long Context Summary:")
    print(response.content)


def gemini_streaming_example():
    """Example of streaming with Gemini."""
    client = Client()
    
    print("Streaming response from Gemini:")
    stream = client.completion.create(
        messages=[
            {"role": "user", "content": "Write a short story about a robot learning to paint"}
        ],
        model="google/gemini-1.5-flash",
        stream=True,
        max_tokens=300
    )
    
    for chunk in stream:
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")


def gemini_with_safety_settings():
    """Example using Gemini with custom safety settings."""
    client = Client()
    
    # Gemini-specific safety settings
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
    ]
    
    response = client.completion.create(
        messages=[
            {"role": "user", "content": "What are the benefits of renewable energy?"}
        ],
        model="google/gemini-1.5-pro",
        safety_settings=safety_settings  # Gemini-specific parameter
    )
    
    print(f"Response with safety settings:\n{response.content}")


async def gemini_async_example():
    """Async example with Gemini."""
    client = Client()
    
    # Make multiple async requests
    tasks = [
        client.completion.acreate(
            messages=[{"role": "user", "content": f"What is {topic}?"}],
            model="google/gemini-1.5-flash-8b"  # Fastest model for quick responses
        )
        for topic in ["quantum computing", "blockchain", "machine learning"]
    ]
    
    responses = await asyncio.gather(*tasks)
    
    for topic, response in zip(["quantum computing", "blockchain", "machine learning"], responses):
        print(f"\n{topic.title()}:")
        print(response.content[:150] + "...")


def gemini_25_flash_features():
    """Demonstrate Gemini 2.5 Flash's unique features."""
    client = Client()
    
    print("Gemini 2.5 Flash - Latest Features Demo:\n")
    
    # 1. Large output capability (65k tokens)
    print("1. Testing large output capability:")
    response = client.completion.create(
        messages=[
            {"role": "user", "content": "Write a detailed guide on Python decorators with multiple examples"}
        ],
        model="google/gemini-2.5-flash",
        max_tokens=2000  # Can go up to 65536
    )
    print(f"Response length: {len(response.content)} characters")
    print(f"First 200 chars: {response.content[:200]}...\n")
    
    # 2. Code execution capability
    print("2. Code execution (if supported by API):")
    response = client.completion.create(
        messages=[
            {"role": "user", "content": "Write and explain a Python function to find prime numbers up to 100"}
        ],
        model="google/gemini-2.5-flash",
        max_tokens=1000
    )
    print(f"Code example: {response.content[:300]}...\n")
    
    # 3. Multimodal capabilities
    print("3. Multimodal support:")
    print("   - Text: ✓")
    print("   - Images: ✓") 
    print("   - Video: ✓")
    print("   - Audio: ✓")
    print("   - Note: Pass media as base64 encoded data in content\n")
    
    # 4. Advanced features
    print("4. Advanced capabilities:")
    print("   - Caching: Supported")
    print("   - Function calling: Supported")
    print("   - Search grounding: Supported")
    print("   - Structured outputs: Supported")
    print("   - Thinking mode: Supported")
    print("   - Batch mode: Supported")
    

def cost_comparison_example():
    """Compare costs across different providers for the same task."""
    client = Client()
    
    models = [
        "google/gemini-1.5-flash-8b",   # Google - Cheapest
        "openai/gpt-3.5-turbo",         # OpenAI - Cheap
        "anthropic/claude-3-5-haiku-20241022",  # Anthropic - Cheap
        "google/gemini-1.5-pro",        # Google - Premium
        "openai/gpt-4o",                # OpenAI - Premium
    ]
    
    prompt = "Explain the difference between machine learning and deep learning"
    
    print("Cost Comparison for Different Models:\n")
    print(f"{'Model':<40} {'Tokens':<10} {'Cost':<15} {'Cost/1K tokens'}")
    print("-" * 80)
    
    for model in models:
        try:
            response = client.completion.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=150
            )
            
            if response.usage:
                cost_per_1k = (response.usage.estimated_cost / response.usage.total_tokens) * 1000 if response.usage.total_tokens > 0 else 0
                print(f"{model:<40} {response.usage.total_tokens:<10} ${response.usage.estimated_cost:<14.6f} ${cost_per_1k:.6f}")
        except Exception as e:
            print(f"{model:<40} Error: {str(e)[:30]}...")


if __name__ == "__main__":
    print("=== Google Gemini Examples with JustLLMs ===\n")
    
    print("1. Basic Gemini Example:")
    basic_gemini_example()
    
    print("\n2. Gemini Model Comparison:")
    gemini_model_comparison()
    
    print("\n3. Gemini 2.5 Flash Features:")
    gemini_25_flash_features()
    
    print("\n4. Long Context Example:")
    gemini_long_context_example()
    
    print("\n5. Streaming Example:")
    gemini_streaming_example()
    
    print("\n6. Safety Settings Example:")
    gemini_with_safety_settings()
    
    print("\n7. Async Example:")
    asyncio.run(gemini_async_example())
    
    print("\n8. Cost Comparison:")
    cost_comparison_example()