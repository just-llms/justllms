# JustLLMs

A production-ready Python library for multi-provider LLM management with a unified API.

[![CI](https://github.com/just-llms/justllms/actions/workflows/ci.yml/badge.svg)](https://github.com/just-llms/justllms/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/justllms.svg)](https://pypi.org/project/justllms/)
[![Downloads](https://pepy.tech/badge/justllms)](https://pepy.tech/project/justllms)
[![GitHub issues](https://img.shields.io/github/issues/just-llms/justllms.svg)](https://github.com/just-llms/justllms/issues)
## Why JustLLMs?

Managing multiple LLM providers is complex. You need to handle different APIs, manage authentication, implement tool calling differently for each provider, and ensure reliability. JustLLMs solves these challenges by providing:

- **Unified Interface**: One API for all providers (OpenAI, Anthropic, Google, Azure, xAI, DeepSeek, Ollama)
- **Provider-Agnostic Tool Calling**: Define tools once, use them with any provider
- **Automatic Fallbacks**: Built-in reliability with configurable fallback providers
- **Side-by-Side Comparison**: Interactive CLI to compare multiple models simultaneously

## Installation

```bash
pip install justllms
```

## Quick Start

```python
from justllms import JustLLM

# Initialize with your API keys
client = JustLLM({
    "providers": {
        "openai": {"api_key": "your-openai-key"},
        "google": {"api_key": "your-google-key"},
        "anthropic": {"api_key": "your-anthropic-key"}
    }
})

# Simple completion - uses configured fallback or first available provider
response = client.completion.create(
    messages=[{"role": "user", "content": "Explain quantum computing briefly"}]
)
print(response.content)
```

## Core Features

### Multi-Provider Support
Connect to all major LLM providers with a single, consistent interface:
- **OpenAI** (GPT-5, GPT-4, etc.)
- **Google** (Gemini 2.5, etc)
- **Anthropic** (Claude 4, Claude 3.5 models)
- **Azure OpenAI** (with deployment mapping)
- **xAI Grok**, **DeepSeek**
- **Ollama** (local Llama/Mistral/phi models hosted on your machine)

```python
# Switch between providers seamlessly
client = JustLLM({
    "providers": {
        "openai": {"api_key": "your-key"},
        "google": {"api_key": "your-key"},
        "anthropic": {"api_key": "your-key"},
        "ollama": {"base_url": "http://localhost:11434"}
    }
})

# Explicitly specify provider and model
response1 = client.completion.create(
    messages=[{"role": "user", "content": "Explain AI"}],
    model="openai/gpt-4o"  # Format: "provider/model"
)
```

Ollama runs locally and requires no API key. Set `OLLAMA_API_BASE` (defaults to
`http://localhost:11434`) and JustLLMs automatically discovers every installed
model via the Ollama `/api/tags` endpoint.

### Provider-Agnostic Tool Calling

Define tools once, use them with any provider - no need to learn different tool calling APIs:

```python
from justllms import JustLLM, tool

@tool
def get_weather(location: str) -> dict:
    """Get weather for a location."""
    return {"temperature": 22, "condition": "sunny"}

# Works with OpenAI, Anthropic, Google - same code!
response = client.completion.create(
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[get_weather],
    provider="openai",  # or "anthropic", "google"
    execute_tools=True
)
```

Native tools support:
```python
from justllms import GoogleSearch, GoogleCodeExecution

# Server-side Google Search and Python execution
response = client.completion.create(
    messages=[{"role": "user", "content": "Latest AI news and calculate 2^10"}],
    tools=[GoogleSearch(), GoogleCodeExecution()],
    provider="google"
)
```

### Automatic Fallbacks
Configure fallback providers and models for reliability:

```python
client = JustLLM({
    "providers": {
        "openai": {"api_key": "your-key"},
        "anthropic": {"api_key": "your-key"}
    },
    "routing": {
        "fallback_provider": "anthropic",
        "fallback_model": "claude-3-5-sonnet-20241022"
    }
})

# If no model specified, uses fallback
response = client.completion.create(
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Side-by-Side Model Comparison

Compare multiple LLM providers and models simultaneously with our interactive SXS (Side-by-Side) comparison tool. Perfect for evaluating model performance, testing prompts, and making informed decisions about which models to use.

### Features
- **Interactive CLI**: Select providers and models using checkbox interface
- **Parallel Execution**: All models run simultaneously for fair comparison
- **Real-time Results**: Live display with loading animation until all models complete
- **Comprehensive Metrics**: Compare latency, token usage, response quality and costs across models
- **Multiple Providers**: Test OpenAI, Google, Anthropic, xAI, DeepSeek models side-by-side

### Usage

```bash
# Run the interactive SXS comparison
justllms sxs
```

The tool will guide you through:

1. **Provider Selection**: Choose which LLM providers to compare
2. **Model Selection**: Pick specific models from each provider  
3. **Prompt Input**: Enter your test prompt
4. **Real-time Comparison**: View all responses and metrics simultaneously

### Example Output
```
================================================================================
Prompt: Which programming language is better for beginners: Python or JavaScript?
================================================================================

┌─ openai/gpt-5          ─────────────────────────────────────────────────────┐
│ Python is generally better for beginners due to its clean, readable syntax │
│ that resembles natural language. It has fewer confusing concepts like       │
│ hoisting or prototypes, excellent learning resources, and is widely used    │
│ in education. Python's "batteries included" philosophy means beginners can  │
│ accomplish tasks without learning complex setups, making it ideal for       │
│ building confidence early in programming.                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ google/gemini-2.5-pro ─────────────────────────────────────────────────────┐
│ JavaScript has advantages for beginners because it runs everywhere - in     │
│ browsers, servers, and mobile apps. You can see immediate visual results    │
│ when building web pages, which is motivating. The job market heavily favors │
│ JavaScript developers, and modern frameworks make it powerful. While syntax │
│ can be tricky, the instant feedback and versatility make JavaScript a       │
│ practical first language for aspiring developers.                           │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
Metrics Summary:

| Model                   |  Status   | Latency (s) | Tokens | Cost ($) |
|-------------------------|-----------|-------------|--------|----------|
| openai/gpt-5            | ✓ Success |        5.69 |    715 |   0.0000 |
| google/gemini-2.5-pro   | ✓ Success |       8.50 |    868 |   0.0003  |
```

### Streaming Support
Stream responses in real-time for interactive applications with a **provider-agnostic API** - no need to learn different SDKs or streaming implementations. The same code works across OpenAI, Google Gemini, and Azure OpenAI:

```python
# Same streaming code works for ANY supported provider!
response = client.completion.create(
    messages=[{"role": "user", "content": "Write a story about AI"}],
    provider="google",  # or "openai", "azure_openai"
    model="gemini-2.5-flash",
    stream=True
)

# Identical iteration pattern across all providers
for chunk in response:
    if chunk.content:
        print(chunk.content, end="", flush=True)

# Get final response with usage stats and cost estimation
final = response.get_final_response()
print(f"\n\nTokens used: {final.usage.total_tokens}")
print(f"Cost: ${final.usage.estimated_cost:.6f}")
```

**No SDK hassle:**
- ❌ Don't learn OpenAI's `stream=True` SSE format
- ❌ Don't learn Gemini's `generate_content_stream()` method
- ❌ Don't learn Ollama's newline-delimited JSON streaming
- ❌ Don't handle different chunk formats per provider
- ✅ **One API, all providers** - just set `stream=True`

## Tool Calling (Function Calling)

JustLLMs provides a **provider-agnostic tool calling API** that works seamlessly across OpenAI, Anthropic, and Google Gemini. Define tools once, use them everywhere.

### Basic Tool Calling

Define tools using the `@tool` decorator:

```python
from justllms import JustLLM, tool

# Define a tool with the @tool decorator
@tool
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get the current weather for a location.

    Args:
        location: The city and state, e.g., "San Francisco, CA"
        unit: Temperature unit (celsius or fahrenheit)

    Returns:
        Weather information including temperature and conditions
    """
    # Your implementation here
    return {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "condition": "sunny"
    }

# Use the tool with any provider
client = JustLLM()

response = client.completion.create(
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[get_weather],
    provider="openai",  # or "anthropic", "google"
    execute_tools=True  # Automatically execute tools
)

print(response.content)
# "The weather in Paris is currently 22°C and sunny."
```

### Provider-Agnostic Support

**The same tool code works across all providers** - no need to adapt your tools for different APIs:

```python
# Works with OpenAI
response = client.completion.create(
    messages=[{"role": "user", "content": "What's the weather in London?"}],
    tools=[get_weather],
    provider="openai",
    execute_tools=True
)

# Same code works with Anthropic Claude
response = client.completion.create(
    messages=[{"role": "user", "content": "What's the weather in London?"}],
    tools=[get_weather],
    provider="anthropic",
    execute_tools=True
)

# Same code works with Google Gemini
response = client.completion.create(
    messages=[{"role": "user", "content": "What's the weather in London?"}],
    tools=[get_weather],
    provider="google",
    execute_tools=True
)
```

### Multi-Tool Support

Define and use multiple tools together:

```python
@tool
def get_weather(location: str) -> dict:
    """Get weather for a location."""
    return {"temperature": 22, "condition": "sunny"}

@tool
def get_time(timezone: str) -> str:
    """Get current time in a timezone."""
    return "2024-01-15 14:30:00"

@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)  # Note: Use safely in production

# Use all tools together
response = client.completion.create(
    messages=[{
        "role": "user",
        "content": "What's the weather in Paris, the time in EST, and what's 15 * 23?"
    }],
    tools=[get_weather, get_time, calculate],
    execute_tools=True
)
```


## 🏆 Comparison with Alternatives

| Feature | JustLLMs | LangChain | LiteLLM | OpenAI SDK |
|---------|----------|-----------|---------|------------|
| **Package Size** | Minimal | ~50MB | ~5MB | ~1MB |
| **Setup Complexity** | Simple config | Complex chains | Medium | Simple |
| **Multi-Provider** | ✅ 7+ providers | ✅ Many integrations | ✅ 100+ providers | ❌ OpenAI only |
| **Unified API** | ✅ Single interface | ⚠️ Different patterns | ⚠️ Provider-specific | ❌ OpenAI only |
| **Tool Calling** | ✅ Provider-agnostic | ⚠️ Manual handling | ⚠️ Provider-specific | ⚠️ OpenAI only |
| **Native Tools** | ✅ Google Search/Code | ❌ None | ❌ None | ❌ None |
| **Side-by-Side Comparison** | ✅ Interactive CLI tool | ❌ None | ❌ None | ❌ None |
| **Automatic Fallbacks** | ✅ Built-in | ❌ Manual | ⚠️ Basic | ❌ None |
| **Production Ready** | ✅ Out of the box | ⚠️ Requires setup | ✅ Minimal setup | ⚠️ Basic features |


[![Star History Chart](https://api.star-history.com/svg?repos=just-llms/justllms&type=Date)](https://www.star-history.com/#just-llms/justllms&Date)
