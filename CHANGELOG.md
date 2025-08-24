# Changelog

All notable changes to JustLLMs will be documented in this file.

## [1.3.0] - 2025-08-11

### Added - Major Model Updates 🚀
- **Latest 2025 Models**: Comprehensive update with all newest models across providers
  - **OpenAI**: GPT-5, GPT-5 Mini, GPT-4.1, GPT-4.1 Nano, O1, O4-Mini, GPT-OSS-120b
  - **Google**: Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 2.5 Flash-Lite (latest generation)
  - **Anthropic**: Claude Opus 4.1, Claude Sonnet 4, Claude Haiku 3.5 (new 4.x series)
  - **Grok**: Grok 4, Grok 4 Heavy, Grok 3 Speedy, Grok 3 Mini variants
  - **DeepSeek**: Enhanced models with cache-hit pricing variants (75% discount support)
  - **Azure OpenAI**: GPT-5 series, O3, O4-Mini, and latest deployment mappings

### Enhanced Routing Intelligence 🎯
- **5-Tier Quality System**: Upgraded from 4-tier to 5-tier quality classification
  - Tier 5: Premium models (GPT-5, Claude Opus 4.1, Grok 4)
  - Enhanced model capability matching and selection logic
- **Task-Based Routing Updates**: Optimized preferences for new models
  - Code tasks: Prioritizes Gemini 2.5 Pro, GPT-5, DeepSeek Reasoner
  - Analysis: Prioritizes O1, Claude Opus 4.1, reasoning specialists
  - Creative: Prioritizes Claude Opus 4.1, Grok 4 for creative responses
  - Simple: Cost-optimized with Flash-Lite, GPT-4.1 Nano, DeepSeek Chat
- **Latency Optimization**: Updated scoring for nano/lite models, provider-specific adjustments
- **Provider Performance**: Added DeepSeek (0.7x fastest), Grok (1.1x), Azure OpenAI (0.85x) multipliers

### New Model Capabilities 🎨
- **Extended Context Windows**: Up to 2M tokens (Gemini 1.5 Pro), 1M tokens (GPT-4.1, Gemini 2.5)
- **Multimodal Enhancements**: Vision support across Grok 4, Gemini 2.5, Claude 4.x series
- **Reasoning Models**: Specialized O1, O4-Mini for complex analytical tasks
- **Cost Efficiency**: Ultra-affordable options (DeepSeek cached, Flash-Lite, GPT-4.1 Nano)
- **Tool Chaining**: Enhanced function calling and tool integration (GPT-5, Grok 4)

### Updated Model Specifications 📊
- **Accurate Pricing**: 2025 pricing for all models including cache discounts
- **Context Lengths**: Verified context windows and output token limits  
- **Capability Tags**: Enhanced tagging system (flagship, reasoning, multimodal, cost-efficient)
- **Provider-Specific**: Azure deployment name mapping for new models

### Configuration Enhancements ⚙️
- **Backward Compatibility**: Legacy models retained alongside new variants
- **Smart Defaults**: Intelligent routing continues to select cost-optimal models
- **Model Validation**: Enhanced model availability checking across providers

## [1.2.0] - 2025-08-10

### Added  
- **Enhanced Multi-Provider Support**: Expanded provider ecosystem with improved routing
- **Advanced Analytics**: Enhanced cost tracking and performance monitoring
- **Conversation Management**: Persistent conversation storage and context management
- **Business Rule Validation**: Enterprise-grade content filtering and compliance

### New Components
- **Enhanced Routing**: Improved intelligent routing strategies
- **Analytics Dashboard**: Comprehensive usage and cost analysis
- **Conversation System**: Full lifecycle conversation management
- **Validation Engine**: Content filtering and business rules

### Examples Added
- `examples/basic_completions.py`: Basic completion workflows
- `examples/conversation_flow.py`: Conversation management examples
- `examples/cost_tracking.py`: Cost analysis demonstrations
- `examples/routing.py`: Intelligent routing examples

### Configuration Enhancements
- **Provider Management**: Simplified provider configuration
- **Routing Strategies**: Cost, latency, and quality optimization
- **Analytics Configuration**: Comprehensive monitoring setup

### API Enhancements
- **client.completion.create()**: Enhanced completion interface
- **client.conversations**: Full conversation management
- **client.analytics**: Built-in analytics dashboard

### Documentation
- **README.md**: Enhanced core features documentation
- **Feature Comparison**: Updated comparison with other libraries
- **Configuration Examples**: Complete setup guides for all features

### Technical Improvements
- **Performance Optimization**: Improved response times and efficiency
- **Error Handling**: Enhanced error handling and fallback mechanisms
- **Async Support**: Full async/await support across all operations
- **Caching**: Smart caching for improved performance

## [1.1.0] - 2025-08-08

### Added
- **Enhanced Provider Support**: Improved compatibility and error handling across providers
- **Conversational Flow Updates**: Better conversation state management and context handling

### Fixed
- **Sync Conversational Flow**: Fixed synchronous conversation handling for better reliability
- **Provider Integration**: Improved provider initialization and configuration validation

## [1.0.1] - 2025-08-08

### Fixed
- **Azure OpenAI Provider**: Fixed provider name registration issue where "azure" configuration key was not being recognized at runtime
- **Provider Name Handling**: Provider now correctly uses the configuration key name instead of hardcoded "azure_openai"

### Changed
- Azure provider models now use "azure" as the default provider identifier for consistency

## [1.0.0] - 2025-08-07

### Added
- **Multi-Provider Support**: Support for OpenAI, Azure OpenAI, Google Gemini, Anthropic Claude, DeepSeek, and xAI Grok
- **Intelligent Routing**: Cost-optimized, latency-optimized, quality-optimized, and task-based routing strategies
- **Conversation Management**: Full conversation lifecycle with context management, auto-save, and export capabilities
- **Advanced Analytics**: Comprehensive reporting with CSV and PDF export, cross-provider metrics, and cost tracking
- **Business Rule Validation**: Enterprise content filtering with customizable rules
- **Production Streaming**: Real-time token streaming with proper chunk handling for all providers
- **Smart Caching**: Intelligent response caching with multiple backend support (Memory, Redis, Disk)
- **Health Monitoring**: Provider health checking with automatic failover
- **Error Handling**: Robust retry logic with exponential backoff
- **Configuration Management**: Flexible configuration system with validation

### Features
- **Cost Intelligence**: Automatic cost optimization and detailed cost tracking per provider/model
- **Context Window Management**: Intelligent context handling with truncation and summarization strategies
- **Export Capabilities**: Export conversations and analytics in JSON, Markdown, TXT, CSV, and PDF formats
- **Async Support**: Full async/await support for high-performance applications
- **Function Calling**: Support for function calling across compatible providers
- **Vision Support**: Multi-modal support for image processing with compatible models
- **Enterprise Ready**: Business rule validation, content filtering, and compliance features

### Technical
- **Streaming Fixed**: Fixed Azure OpenAI streaming to properly handle delta objects
- **Provider Abstractions**: Unified interface across all LLM providers
- **Plugin Architecture**: Extensible architecture for adding new providers and features
- **Type Safety**: Full type hints and validation using Pydantic models
- **Comprehensive Testing**: Extensive test coverage for all features

### Documentation
- **Feature Guide**: Comprehensive feature documentation with examples
- **API Documentation**: Complete API reference with Sphinx
- **Examples**: Ready-to-run examples for common use cases
- **Configuration Guide**: Detailed configuration documentation

## Developer Notes

This is the initial stable release of JustLLMs, providing enterprise-grade LLM orchestration with intelligent routing, comprehensive analytics, and production-ready features.

### Breaking Changes
- None (initial release)

### Migration Guide
- None (initial release)

### Known Issues
- None currently reported

### Contributors
- Core development team
- Community contributors

## Roadmap

### Next Release (1.4.0)
- **Advanced Caching**: Redis-backed intelligent caching with semantic similarity
- **Function Calling Enhancements**: Enhanced function calling support with tool chaining
- **Additional Vector Stores**: Support for more vector databases (Weaviate, Qdrant, Elasticsearch)
- **Advanced RAG**: Semantic chunking strategies and hybrid search capabilities
- **Performance Monitoring**: Real-time latency tracking and provider health monitoring

### Future Plans
- **Web-based Analytics Dashboard**: Visual insights and real-time monitoring
- **Advanced Conversation Analytics**: Sentiment analysis, topic modeling, conversation scoring
- **Custom Model Fine-tuning Integration**: Train and deploy custom models seamlessly
- **Enterprise SSO Support**: OAuth, SAML, and directory integration
- **Enhanced Compliance Tools**: SOC 2, ISO 27001 audit trails
- **Multi-region Deployment**: Automatic geographic routing for performance