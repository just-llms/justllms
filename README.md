# JustLLMs

A production-ready Python library for multi-provider LLM management with a unified API.

[![CI](https://github.com/just-llms/justllms/actions/workflows/ci.yml/badge.svg)](https://github.com/just-llms/justllms/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/justllms.svg)](https://pypi.org/project/justllms/)
[![Downloads](https://pepy.tech/badge/justllms)](https://pepy.tech/project/justllms)
[![GitHub issues](https://img.shields.io/github/issues/just-llms/justllms.svg)](https://github.com/just-llms/justllms/issues)
## Why JustLLMs?

Managing multiple LLM providers is complex. You need to handle different APIs, manage authentication, and ensure reliability. JustLLMs solves these challenges by providing a unified interface across all major providers with automatic fallbacks and consistent error handling.

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


## 🏆 Comparison with Alternatives

| Feature | JustLLMs | LangChain | LiteLLM | OpenAI SDK |
|---------|----------|-----------|---------|------------|
| **Package Size** | Minimal | ~50MB | ~5MB | ~1MB |
| **Setup Complexity** | Simple config | Complex chains | Medium | Simple |
| **Multi-Provider** | ✅ 7+ providers | ✅ Many integrations | ✅ 100+ providers | ❌ OpenAI only |
| **Unified API** | ✅ Single interface | ⚠️ Different patterns | ⚠️ Provider-specific | ❌ OpenAI only |
| **Side-by-Side Comparison** | ✅ Interactive CLI tool | ❌ None | ❌ None | ❌ None |
| **Automatic Fallbacks** | ✅ Built-in | ❌ Manual | ⚠️ Basic | ❌ None |
| **Production Ready** | ✅ Out of the box | ⚠️ Requires setup | ✅ Minimal setup | ⚠️ Basic features |

## Provider-Specific Parameters

JustLLMs supports common generation parameters across all providers, plus provider-specific configurations:

### Common Parameters (All Providers)

These parameters work across OpenAI, Gemini, Anthropic, and other providers:

```python
response = client.completion.create(
    messages=[{"role": "user", "content": "Hello"}],
    # Common parameters
    temperature=0.7,        # 0.0-2.0: Controls randomness
    top_p=0.9,             # 0.0-1.0: Nucleus sampling
    top_k=40,              # Integer: Top-k sampling (Gemini only)
    max_tokens=1024,       # Maximum tokens to generate
    stop=["END"],          # Stop sequence(s)
    n=1,                   # Number of completions (OpenAI only)
    presence_penalty=0.1,  # -2.0 to 2.0: Penalize new topics
    frequency_penalty=0.2  # -2.0 to 2.0: Penalize repetition
)
```

### Gemini-Specific Parameters

Use `generation_config` for Gemini-only features:

```python
response = client.completion.create(
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    provider="google",
    model="gemini-2.5-flash",
    # Common parameters
    temperature=0.7,
    top_k=40,
    max_tokens=1024,
    # Gemini-specific configuration
    generation_config={
        "candidateCount": 2,                    # Generate multiple responses
        "responseMimeType": "application/json", # JSON output
        "responseSchema": {...},                # Structured output schema
        "thinkingConfig": {                     # Control thinking budget
            "thinkingBudget": 100               # 0-24000 tokens
        }
    }
)

# Access multiple candidates when candidateCount > 1
print(f"Candidate 1: {response.choices[0].message.content}")
print(f"Candidate 2: {response.choices[1].message.content}")
```

**Notes:**
- Common parameters (`temperature`, `top_k`, etc.) should be set at the top level. The `generation_config` dict is for Gemini-exclusive features.
- If a parameter is specified in both places, the top-level value takes precedence.
- When `candidateCount > 1`, all candidates are returned in `response.choices[]` with proper indices.

### OpenAI-Specific Parameters

OpenAI parameters are passed directly:

```python
response = client.completion.create(
    messages=[{"role": "user", "content": "Hello"}],
    provider="openai",
    model="gpt-4o",
    # Common parameters
    temperature=0.7,
    max_tokens=100,
    n=1,
    presence_penalty=0.1,
    frequency_penalty=0.2
)
```

**Note:** `top_k` is not supported by OpenAI and will be silently ignored. Use `generation_config` only with Gemini.

## Production Configuration

For production deployments:

```python
production_config = {
    "providers": {
        "azure_openai": {
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "resource_name": "my-enterprise-resource",
            "deployment_mapping": {
                "gpt-4": "my-gpt4-deployment",
                "gpt-3.5-turbo": "my-gpt35-deployment"
            }
        },
        "anthropic": {"api_key": os.getenv("ANTHROPIC_KEY")},
        "google": {"api_key": os.getenv("GOOGLE_KEY")},
        "ollama": {
            "base_url": os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
            "enabled": True,
        }
    },
    "routing": {
        "fallback_provider": "azure_openai",
        "fallback_model": "gpt-3.5-turbo"
    }
}

client = JustLLM(production_config)
```

[![Star History Chart](https://api.star-history.com/svg?repos=just-llms/justllms&type=Date)](https://www.star-history.com/#just-llms/justllms&Date)
