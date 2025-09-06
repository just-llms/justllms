# JustLLMs

A production-ready Python library focused on intelligent LLM routing and multi-provider management.

[![PyPI version](https://badge.fury.io/py/justllms.svg)](https://pypi.org/project/justllms/) [![Downloads](https://pepy.tech/badge/justllms)](https://pepy.tech/project/justllms)

## Why JustLLMs?

Managing multiple LLM providers is complex. You need to handle different APIs, optimize costs, and ensure reliability. JustLLMs solves these challenges by providing a unified interface that automatically routes requests to the best provider based on your criteria—whether that's cost, speed, or quality.

**Key Differentiator**: Advanced cluster-based routing using semantic embeddings to intelligently route queries to optimal models based on research from [AvengersPro](https://arxiv.org/pdf/2508.12631).

## Installation

```bash
pip install justllms
```

**Package size**: Minimal | **Lines of code**: ~7K | **Dependencies**: Production-focused

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
- **xAI Grok**, **DeepSeek**, and more

```python
# Switch between providers seamlessly
client = JustLLM({
    "providers": {
        "openai": {"api_key": "your-key"},
        "google": {"api_key": "your-key"},
        "anthropic": {"api_key": "your-key"}
    }
})

# Same interface, different providers automatically chosen
response1 = client.completion.create(
    messages=[{"role": "user", "content": "Explain AI"}],
    provider="openai"  # Force specific provider
)

response2 = client.completion.create(
    messages=[{"role": "user", "content": "Explain AI"}]
    # Auto-routes to best provider based on your strategy
)
```

### Intelligent Routing
**The game-changing feature that sets JustLLMs apart.** Instead of manually choosing models, let our intelligent routing engine automatically select the optimal provider and model for each request based on your priorities.

#### Available Strategies

**🆕 Cluster-Based Routing** - *AI-Powered Query Analysis*
Our most advanced routing strategy uses machine learning to analyze query semantics and route to the optimal model based on similarity to training data. Achieves **+7% accuracy improvement** and **-27% cost reduction** compared to single-model approaches.

```python
# Cluster-based routing (recommended for production)
client = JustLLM({
    "providers": {...},
    "routing": {"strategy": "cluster"}
})
```

*Based on research from [Beyond GPT-5: Making LLMs Cheaper and Better via Performance–Efficiency Optimized Routing](https://arxiv.org/pdf/2508.12631) - AvengersPro framework*

**Traditional Routing Strategies**

```python
# Cost-optimized: Always picks the cheapest option
client = JustLLM({
    "providers": {...},
    "routing": {"strategy": "cost"}
})

# Speed-optimized: Prioritizes fastest response times
client = JustLLM({
    "providers": {...},
    "routing": {"strategy": "latency"}
})

# Quality-optimized: Uses the best models for complex tasks
client = JustLLM({
    "providers": {...},
    "routing": {"strategy": "quality"}
})

# Task-based: Automatically detects query type and routes accordingly
client = JustLLM({
    "providers": {...},
    "routing": {"strategy": "task"}
})
```

#### How Cluster Routing Works
1. **Query Analysis**: Your request is embedded using Qwen3-Embedding-0.6B
2. **Cluster Matching**: Finds the most similar cluster from pre-trained data
3. **Model Selection**: Routes to the best-performing model for that cluster
4. **Fallback**: Falls back to quality-based routing if needed

**Result**: Up to 60% cost reduction while improving accuracy, with automatic failover to backup providers.

### Cost Estimation
Get cost estimates before making requests:

```python
# Estimate costs for different strategies
cost_estimate = client.estimate_cost(
    messages=[{"role": "user", "content": "Explain AI"}],
    strategy="cluster"
)

print(f"Estimated cost: ${cost_estimate.total_cost:.4f}")
print(f"Selected model: {cost_estimate.selected_model}")
print(f"Provider: {cost_estimate.provider}")
```

## Configuration Management
Flexible configuration with environment variable support:

```python
# Environment-based config
import os
client = JustLLM({
    "providers": {
        "openai": {"api_key": os.getenv("OPENAI_API_KEY")},
        "azure_openai": {
            "api_key": os.getenv("AZURE_OPENAI_KEY"),
            "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "resource_name": os.getenv("AZURE_RESOURCE_NAME"),
            "api_version": "2024-12-01-preview"
        }
    }
})

# File-based config
import yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)
client = JustLLM(config)
```

## 🏆 Comparison with Alternatives

| Feature | JustLLMs | LangChain | LiteLLM | OpenAI SDK |
|---------|----------|-----------|---------|------------|
| **Package Size** | Minimal | ~50MB | ~5MB | ~1MB |
| **Setup Complexity** | Simple config | Complex chains | Medium | Simple |
| **Multi-Provider** | ✅ 6+ providers | ✅ Many integrations | ✅ 100+ providers | ❌ OpenAI only |
| **Intelligent Routing** | ✅ Cost/speed/quality/cluster | ❌ Manual only | ⚠️ Basic routing | ❌ None |
| **Cost Optimization** | ✅ Automatic routing | ❌ Manual optimization | ⚠️ Basic cost tracking | ❌ None |
| **Production Ready** | ✅ Out of the box | ⚠️ Requires setup | ✅ Minimal setup | ⚠️ Basic features |

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
        "google": {"api_key": os.getenv("GOOGLE_KEY")}
    },
    "routing": {
        "strategy": "cluster",  # Use intelligent cluster-based routing
        "fallback_provider": "azure_openai",
        "fallback_model": "gpt-3.5-turbo"
    }
}

client = JustLLM(production_config)
```

## Key Differentiators

1. **Cluster-Based Routing**: AI-powered query analysis for optimal model selection
2. **Production Simplicity**: Minimal dependencies, focused feature set
3. **Cost Optimization**: Automatic routing to reduce costs by up to 60%
4. **Unified Interface**: Same API across all providers
5. **Reliability**: Built-in fallback and error handling

## License

MIT License - see [LICENSE](LICENSE) file for details.

[![Star History Chart](https://api.star-history.com/svg?repos=just-llms/justllms&type=Date)](https://www.star-history.com/#just-llms/justllms&Date)