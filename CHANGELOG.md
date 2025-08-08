# Changelog

All notable changes to JustLLMs will be documented in this file.

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

### Next Release (1.1.0)
- Additional provider integrations
- Enhanced analytics dashboards
- Real-time monitoring improvements
- Performance optimizations

### Future Plans
- Web-based analytics dashboard
- Advanced conversation analytics
- Custom model fine-tuning integration
- Enterprise SSO support