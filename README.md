# JustLLMs

A production-ready Python library that simplifies working with multiple Large Language Model providers through intelligent routing, comprehensive analytics, and enterprise-grade features.

[![PyPI version](https://badge.fury.io/py/justllms.svg)](https://pypi.org/project/justllms/) [![Downloads](https://pepy.tech/badge/justllms)](https://pepy.tech/project/justllms)

## Why JustLLMs?

Managing multiple LLM providers is complex. You need to handle different APIs, optimize costs, monitor usage, and ensure reliability. JustLLMs solves these challenges by providing a unified interface that automatically routes requests to the best provider based on your criteria—whether that's cost, speed, or quality.

## Installation

```bash
# Basic installation
pip install justllms

# With PDF export capabilities
pip install "justllms[pdf]"

# All optional dependencies (PDF export, Redis caching, advanced analytics)
pip install "justllms[all]"
```

**Package size**: 1.1MB | **Lines of code**: ~11K | **Dependencies**: Minimal production requirements

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
- **OpenAI** (GPT-5, GPT-4, etc.) <yes, you can use GPT 5 :)>
- **Google** (Gemini 2.5, Gemini 1.5 models)  
- **Anthropic** (Claude 3.5, Claude 3 models)
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

*Based on research from "[Beyond GPT-5: Making LLMs Cheaper and Better via Performance–Efficiency Optimized Routing](https://arxiv.org/pdf/2508.12631)" - AvengersPro framework*

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

### Real-time Streaming
Full streaming support with proper token handling across all providers:

```python
stream = client.completion.create(
    messages=[{"role": "user", "content": "Write a short story"}],
    stream=True
)

for chunk in stream:
    print(chunk.content, end="", flush=True)
```

### Conversation Management
Built-in conversation state management with context preservation:

```python
# Create client
conversation = Conversation(client=client)

# Set system message
conversation.add_system_message("You are a helpful math tutor. Keep answers concise.")

# Turn 1
response = conversation.send("What is 15 + 25?")

# Turn 2 - Context is automatically preserved
response = conversation.send("Now divide that by 8")

# Get conversation stats
history = conversation.get_history()
```
**Conversation Features:**
- **Auto-save**: Persist conversations automatically
- **Context management**: Smart context window handling
- **Export/Import**: JSON, Markdown, and TXT formats
- **Analytics**: Track usage, costs, and performance per conversation
- **Search**: Find conversations by content or metadata

### Smart Caching
Intelligent response caching that dramatically reduces costs and improves response times:

```python
client = JustLLM({
    "providers": {...},
    "caching": {
        "enabled": True,
        "ttl": 3600,  # 1 hour
        "max_size": 1000
    }
})

# First call - cache miss
response1 = client.completion.create(
    messages=[{"role": "user", "content": "What is AI?"}]
)  # ~2 seconds, full cost

# Second call - cache hit
response2 = client.completion.create(
    messages=[{"role": "user", "content": "What is AI?"}]
)  # ~50ms, no cost
```

### Enterprise Analytics
**Comprehensive usage tracking and cost analysis** that gives you complete visibility into your LLM operations. Unlike other solutions that require external tools, JustLLMs provides built-in analytics that finance and engineering teams actually need.

#### What You Get
- **Cross-provider metrics**: Compare performance across providers
- **Cost tracking**: Detailed cost analysis per model/provider
- **Performance insights**: Latency, throughput, success rates
- **Export capabilities**: CSV, PDF with charts
- **Time series analysis**: Usage patterns over time
- **Top models/providers**: Usage and cost rankings

```python
# Generate detailed reports
report = client.analytics.generate_report()
print(f"Total requests: {report.cross_provider_metrics.total_requests}")
print(f"Total cost: ${report.cross_provider_metrics.total_cost:.2f}")
print(f"Fastest provider: {report.cross_provider_metrics.fastest_provider}")
print(f"Cost per request: ${report.cross_provider_metrics.avg_cost_per_request:.4f}")

# Get granular insights
print(f"Cache hit rate: {report.performance_metrics.cache_hit_rate:.1f}%")
print(f"Token efficiency: {report.optimization_suggestions.token_savings:.1f}%")

# Export reports for finance teams
from justllms.analytics.reports import CSVExporter, PDFExporter
csv_exporter = CSVExporter()
csv_exporter.export(report, "monthly_llm_costs.csv")

pdf_exporter = PDFExporter(include_charts=True)
pdf_exporter.export(report, "executive_summary.pdf")
```

**Business Impact**: Teams typically save 40-70% on LLM costs within the first month by identifying usage patterns and optimizing model selection.

### Unified LLM Interface
**Streamlined access to multiple LLM providers** with intelligent routing, comprehensive analytics, and enterprise-grade features for production deployments.

### Business Rule Validation
**Enterprise-grade content filtering and compliance** built for regulated industries. Ensure your LLM applications meet security, privacy, and business requirements without custom development.

#### Compliance Features
- **PII Detection** - Automatically detect and handle social security numbers, credit cards, phone numbers
- **Content Filtering** - Block inappropriate content, profanity, or sensitive topics
- **Custom Business Rules** - Define your own validation logic with regex patterns or custom functions
- **Audit Trail** - Complete logging of all validation actions for compliance reporting

```python
from justllms.validation import ValidationConfig, BusinessRule, RuleType, ValidationAction

client = JustLLM({
    "providers": {...},
    "validation": ValidationConfig(
        enabled=True,
        business_rules=[
            # Block sensitive data patterns
            BusinessRule(
                name="no_ssn",
                type=RuleType.PATTERNS,
                pattern=r"\\b\\d{3}-\\d{2}-\\d{4}\\b",
                action=ValidationAction.BLOCK,
                message="SSN detected - request blocked for privacy"
            ),
            # Content filtering
            BusinessRule(
                name="professional_content",
                type=RuleType.CONTENT_FILTER,
                categories=["hate", "violence", "adult"],
                action=ValidationAction.SANITIZE
            ),
            # Custom business logic
            BusinessRule(
                name="company_policy",
                type=RuleType.CUSTOM,
                validator=lambda content: "competitor" not in content.lower(),
                action=ValidationAction.WARN
            )
        ],
        # Compliance presets
        compliance_mode="GDPR",  # or "HIPAA", "PCI_DSS"
        audit_logging=True
    )
})

# All requests are automatically validated
response = client.completion.create(
    messages=[{"role": "user", "content": "My SSN is 123-45-6789"}]
)
# This request would be blocked and logged for compliance
```

**Regulatory Compliance**: Built-in support for major compliance frameworks saves months of custom security development.

## Advanced Usage

### Async Operations
Full async/await support for high-performance applications:

```python
import asyncio

async def process_batch():
    tasks = []
    for prompt in prompts:
        task = client.completion.acreate(
            messages=[{"role": "user", "content": prompt}]
        )
        tasks.append(task)
    
    responses = await asyncio.gather(*tasks)
    return responses
```

### Error Handling & Reliability
Automatic retries and fallback providers ensure high availability:

```python
client = JustLLM({
    "providers": {...},
    "retry": {
        "max_attempts": 3,
        "backoff_factor": 2,
        "retry_on": ["timeout", "rate_limit", "server_error"]
    }
})

# Automatically retries on failures
try:
    response = client.completion.create(
        messages=[{"role": "user", "content": "Hello"}],
        provider="invalid-provider"  # Will fail and retry
    )
except Exception as e:
    print(f"All retries failed: {e}")
```

### Configuration Management
Flexible configuration with environment variable support:

```python
# Environment-based config
import os
client = JustLLM({
    "providers": {
        "openai": {"api_key": os.getenv("OPENAI_API_KEY")},
        "azure_openai": {
            "api_key": os.getenv("AZURE_OPENAI_KEY"),
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

| Feature | JustLLMs | LangChain | LiteLLM | OpenAI SDK | Haystack |
|---------|----------|-----------|---------|------------|----------|
| **Package Size** | 1.1MB | ~50MB | ~5MB | ~1MB | ~20MB |
| **Setup Complexity** | Simple config | Complex chains | Medium | Simple | Complex |
| **Multi-Provider** | ✅ 6+ providers | ✅ Many integrations | ✅ 100+ providers | ❌ OpenAI only | ✅ Limited LLMs |
| **Intelligent Routing** | ✅ Cost/speed/quality | ❌ Manual only | ⚠️ Basic routing | ❌ None | ❌ Pipeline-based |
| **Built-in Analytics** | ✅ Enterprise-grade | ❌ External tools needed | ⚠️ Basic metrics | ❌ None | ⚠️ Pipeline metrics |
| **Conversation Management** | ✅ Full lifecycle | ⚠️ Memory components | ❌ None | ❌ Manual handling | ✅ Dialog systems |
| **Business Rules** | ✅ Content validation | ❌ Custom implementation | ❌ None | ❌ None | ⚠️ Custom filters |
| **Cost Optimization** | ✅ Automatic routing | ❌ Manual optimization | ⚠️ Basic cost tracking | ❌ None | ❌ None |
| **Streaming Support** | ✅ All providers | ✅ Provider-dependent | ✅ Most providers | ✅ OpenAI only | ⚠️ Limited |
| **Production Ready** | ✅ Out of the box | ⚠️ Requires setup | ✅ Minimal setup | ⚠️ Basic features | ✅ Complex setup |
| **Caching** | ✅ Multi-backend | ⚠️ Custom implementation | ✅ Basic caching | ❌ None | ✅ Document stores |

## Enterprise Configuration

For production deployments with advanced features:

```python
enterprise_config = {
    "providers": {
        "azure_openai": {
            "api_key": os.getenv("AZURE_OPENAI_KEY"),
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
        "strategy": "cost",
        "fallback_provider": "azure_openai",
        "fallback_model": "gpt-3.5-turbo"
    },
    "validation": {
        "enabled": True,
        "business_rules": [
            # PII detection, content filtering, compliance rules
        ]
    },
    "analytics": {
        "enabled": True,
        "track_usage": True,
        "track_performance": True
    },
    "caching": {
        "enabled": True,
        "backend": "redis",
        "ttl": 3600
    },
    "conversations": {
        "backend": "disk",
        "auto_save": True,
        "auto_title": True,
        "max_context_tokens": 8000
    }
}

client = JustLLM(enterprise_config)
```
## License

MIT License - see [LICENSE](LICENSE) file for details.

[![Star History Chart](https://api.star-history.com/svg?repos=just-llms/justllms&type=Date)](https://www.star-history.com/#just-llms/justllms&Date)
