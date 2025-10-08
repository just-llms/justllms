from justllms.tools.adapters.openai import OpenAIToolAdapter


class AzureToolAdapter(OpenAIToolAdapter):
    """Azure OpenAI uses the same function calling format as OpenAI.

    Since Azure OpenAI Service provides OpenAI's models through Azure,
    they use identical API formats for function calling. This adapter
    simply inherits all functionality from the OpenAI adapter.
    """

    pass  # All functionality inherited from OpenAIToolAdapter
