# JustLLMs

A production-ready Python library focused on intelligent LLM routing and multi-provider management.

[![PyPI version](https://badge.fury.io/py/justllms.svg)](https://pypi.org/project/justllms/) [![Downloads](https://pepy.tech/badge/justllms)](https://pepy.tech/project/justllms)

## Why JustLLMs?

Managing multiple LLM providers is complex. You need to handle different APIs, optimize costs, and ensure reliability. JustLLMs solves these challenges by providing a unified interface that automatically routes requests to the best provider based on your criteria—whether that's cost, speed, or quality. By default, JustLLMs uses intelligent cluster-based routing (beta) powered by machine learning to optimize for all three factors simultaneously.

## Installation

```bash
pip install justllms
```

**Package size**: ~113KB | **Lines of code**: ~4.3K | **Dependencies**: Production-focused

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

# Simple completion - automatically routes to best provider
response = client.completion.create(
    messages=[{"role": "user", "content": "Explain quantum computing briefly"}]
)
print(response.content)
```

## Core Features

### Multi-Provider Support
Connect to all major LLM providers with a single, consistent interface:
- **OpenAI** (GPT-5, GPT-4, etc.)
- **Google** (Gemini 2.5, Gemini 1.5 models)  
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

# Same interface, different providers automatically chosen
response1 = client.completion.create(
    messages=[{"role": "user", "content": "Explain AI"}],
    provider="openai",  # Force specific provider
    model="gpt-5"
)
```

Ollama runs locally and requires no API key. Set `OLLAMA_API_BASE` (defaults to
`http://localhost:11434`) and JustLLMs automatically discovers every installed
model via the Ollama `/api/tags` endpoint.

## **Intelligent Routing**
**The game-changing feature that sets JustLLMs apart.** Instead of manually choosing models, let our intelligent routing engine automatically select the optimal provider and model for each request based on your priorities.

#### Available Strategies

**🆕 Cluster-Based Routing (Beta)** - *AI-Powered Query Analysis*
Our most advanced routing strategy uses machine learning to analyze query semantics and route to the optimal model based on similarity to training data. Achieves **+7% accuracy improvement** and **-27% cost reduction** compared to single-model approaches.

```python
# Cluster-based routing (recommended for production)
client = JustLLM({
    "providers": {...},
    "routing": {"strategy": "cluster"}
})
```

*Based on research from [Beyond GPT-5: Making LLMs Cheaper and Better via Performance–Efficiency Optimized Routing](https://arxiv.org/pdf/2508.12631) - AvengersPro framework*

#### How Cluster Routing Works
1. **Query Analysis**: Your request is embedded using Qwen3-Embedding-0.6B
2. **Cluster Matching**: Finds the most similar cluster from pre-trained data
3. **Model Selection**: Routes to the best-performing model for that cluster
4. **Fallback**: Falls back to configured fallback provider/model or first available if cluster routing is unavailable

**Result**: Up to 60% cost reduction while improving accuracy, with automatic failover to backup providers.

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



## 🏆 Comparison with Alternatives

| Feature | JustLLMs | LangChain | LiteLLM | OpenAI SDK |
|---------|----------|-----------|---------|------------|
| **Package Size** | Minimal | ~50MB | ~5MB | ~1MB |
| **Setup Complexity** | Simple config | Complex chains | Medium | Simple |
| **Multi-Provider** | ✅ 7+ providers | ✅ Many integrations | ✅ 100+ providers | ❌ OpenAI only |
| **Intelligent Routing** | ✅ ML-powered cluster routing | ❌ Manual only | ⚠️ Basic routing | ❌ None |
| **Side-by-Side Comparison** | ✅ Interactive CLI tool | ❌ None | ❌ None | ❌ None |
| **Cost Optimization** | ✅ Automatic routing | ❌ Manual optimization | ⚠️ Basic cost tracking | ❌ None |
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
            "api_key": os.getenv("AZURE_OPENAI_KEY"),
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
        "strategy": "cluster",  # Use intelligent cluster-based routing
        "fallback_provider": "azure_openai",
        "fallback_model": "gpt-3.5-turbo"
    }
}

client = JustLLM(production_config)
```

[![Star History Chart](https://api.star-history.com/svg?repos=just-llms/justllms&type=Date)](https://www.star-history.com/#just-llms/justllms&Date)
