# JustLLMs Examples

This directory contains example scripts demonstrating various features and use cases of the JustLLMs library.

## Examples Overview

### 📘 basic_usage.py
**Getting started with JustLLMs**

Demonstrates fundamental features:
- Simple completions with different providers
- Streaming responses
- Async operations
- Cost monitoring
- Different routing strategies

Perfect for first-time users to understand the core API.

```python
from justllms import Client

client = Client()
response = client.completion.create(
    messages=[{"role": "user", "content": "Hello!"}],
    model="gpt-3.5-turbo"
)
```

### 📗 advanced_usage.py
**Advanced features and customization**

Shows advanced capabilities:
- Creating custom providers
- Implementing custom routing strategies
- Semantic caching
- Budget-aware routing
- Multi-turn conversation management
- Parallel processing with rate limiting

Ideal for developers who want to extend JustLLMs or implement complex use cases.

### 📄 config_example.yaml
**Configuration file template**

A comprehensive configuration example showing:
- Provider settings (API keys, timeouts)
- Routing strategies and parameters
- Cache configuration
- Monitoring and logging setup
- Budget controls

Use this as a template for your own configuration files.

### 🌟 gemini_example.py
**Google Gemini-specific examples**

Demonstrates Gemini-specific features:
- Using different Gemini models (1.5 Pro, Flash, Flash-8B)
- Long context capabilities (up to 2M tokens)
- Safety settings configuration
- Cost comparison with other providers
- Streaming and async operations

Perfect for understanding Gemini's unique capabilities and pricing.

## Running the Examples

1. **Set up API keys:**
   ```bash
   export OPENAI_API_KEY=your-key
   export ANTHROPIC_API_KEY=your-key
   export GOOGLE_API_KEY=your-key
   ```

2. **Install JustLLMs:**
   ```bash
   pip install justllms
   # or for development
   pip install -e ..
   ```

3. **Run examples:**
   ```bash
   python basic_usage.py
   python advanced_usage.py
   ```

## Common Use Cases

### Cost-Conscious Development
```python
# Use cost-optimized routing
client = Client(config={"routing": {"strategy": "cost"}})
```

### High-Quality Outputs
```python
# Use quality-optimized routing
client = Client(config={"routing": {"strategy": "quality"}})
```

### Real-time Applications
```python
# Use latency-optimized routing
client = Client(config={"routing": {"strategy": "latency"}})
```

### Budget Management
```python
# Set spending limits
client = Client(config={
    "monitoring": {
        "cost_tracking": {
            "budget_daily": 10.0,
            "budget_per_request": 0.10
        }
    }
})
```

## Error Handling

JustLLMs provides specific exceptions for different error scenarios:

```python
from justllms import Client, ProviderError, RateLimitError, ValidationError

try:
    response = client.completion.create(messages=[...])
except ValidationError as e:
    print(f"Invalid input: {e}")
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after}s")
except ProviderError as e:
    print(f"Provider error: {e.provider} - {e.message}")
```

## Tips

1. **Enable caching** for development to reduce API calls
2. **Use streaming** for long responses to improve UX
3. **Monitor costs** regularly using the built-in tracking
4. **Configure fallbacks** for production reliability
5. **Implement custom strategies** for domain-specific routing

## Need Help?

- Check the [main documentation](../README.md)
- Explore the [API reference](../docs/api.md)
- Submit issues on [GitHub](https://github.com/yourusername/justllms/issues)