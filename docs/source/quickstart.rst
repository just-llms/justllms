Quick Start Guide
=================

This guide will help you get started with JustLLMs in just a few minutes.

Installation
------------

Install JustLLMs using pip:

.. code-block:: bash

   pip install justllms

For development installation:

.. code-block:: bash

   git clone https://github.com/yourusername/justllms.git
   cd justllms
   pip install -e ".[dev]"

Setting Up API Keys
-------------------

JustLLMs supports multiple LLM providers. Set up your API keys as environment variables:

.. code-block:: bash

   export OPENAI_API_KEY=your-openai-key
   export ANTHROPIC_API_KEY=your-anthropic-key
   export GOOGLE_API_KEY=your-google-key

Basic Usage
-----------

Simple Completion
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from justllms import Client

   # Initialize client
   client = Client()

   # Create a completion
   response = client.completion.create(
       messages=[{"role": "user", "content": "What is the capital of France?"}],
       model="gpt-3.5-turbo"
   )

   print(f"Response: {response.content}")
   print(f"Provider: {response.provider}")
   print(f"Cost: ${response.usage.estimated_cost:.4f}")

Using Different Providers
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Explicitly specify provider/model
   response = client.completion.create(
       messages=[{"role": "user", "content": "Explain quantum computing"}],
       model="anthropic/claude-3-5-sonnet-20241022"
   )

   # Let the router choose based on strategy
   response = client.completion.create(
       messages=[{"role": "user", "content": "What is 2+2?"}],
       # No model specified - router will choose
   )

Streaming Responses
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   stream = client.completion.create(
       messages=[{"role": "user", "content": "Write a short story"}],
       model="gpt-3.5-turbo",
       stream=True,
       max_tokens=200
   )

   for chunk in stream:
       if chunk.content:
           print(chunk.content, end="", flush=True)

Async Operations
~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio

   async def main():
       response = await client.completion.acreate(
           messages=[{"role": "user", "content": "Hello!"}],
           model="gpt-3.5-turbo"
       )
       print(response.content)

   asyncio.run(main())

Error Handling
--------------

JustLLMs provides specific exceptions for different error scenarios:

.. code-block:: python

   from justllms import Client, ProviderError, RateLimitError, ValidationError

   try:
       response = client.completion.create(
           messages=[{"role": "user", "content": "Hello"}],
           model="gpt-3.5-turbo"
       )
   except ValidationError as e:
       print(f"Invalid input: {e.message}")
       print(f"Field: {e.field}, Value: {e.value}")
   except RateLimitError as e:
       print(f"Rate limited by {e.provider}")
       if e.retry_after:
           print(f"Retry after {e.retry_after} seconds")
   except ProviderError as e:
       print(f"Provider {e.provider} error: {e.message}")
       print(f"Status code: {e.status_code}")

Configuration
-------------

Using a Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a ``config.yaml`` file:

.. code-block:: yaml

   providers:
     openai:
       enabled: true
       api_key: ${OPENAI_API_KEY}
     
   routing:
     strategy: "cost"
     
   cache:
     enabled: true
     backend: "memory"
     
   monitoring:
     cost_tracking:
       budget_daily: 10.0

Load and use the configuration:

.. code-block:: python

   from justllms import Client, load_config

   config = load_config("config.yaml")
   client = Client(config=config)

Programmatic Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   client = Client(
       config={
           "routing": {
               "strategy": "quality",
               "strategy_configs": {
                   "quality": {
                       "min_quality_tier": "advanced"
                   }
               }
           },
           "cache": {
               "enabled": True,
               "ttl": 3600
           }
       }
   )

Next Steps
----------

- Explore :doc:`providers` to learn about supported LLM providers
- Read about :doc:`routing` strategies for optimal model selection
- Set up :doc:`monitoring` to track usage and costs
- Check out :doc:`examples` for more use cases