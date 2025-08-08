Client API Reference
====================

.. module:: justllms

Client
------

.. autoclass:: Client
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

Completion
----------

.. autoclass:: justllms.core.completion.Completion
   :members:
   :undoc-members:
   :show-inheritance:

CompletionResponse
------------------

.. autoclass:: justllms.core.completion.CompletionResponse
   :members:
   :undoc-members:
   :show-inheritance:

Models
------

Message
~~~~~~~

.. autoclass:: justllms.core.models.Message
   :members:
   :undoc-members:
   :show-inheritance:

Role
~~~~

.. autoclass:: justllms.core.models.Role
   :members:
   :undoc-members:
   :show-inheritance:

Usage
~~~~~

.. autoclass:: justllms.core.models.Usage
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

Config
~~~~~~

.. autoclass:: justllms.config.Config
   :members:
   :undoc-members:
   :show-inheritance:

load_config
~~~~~~~~~~~

.. autofunction:: justllms.config.load_config

Examples
--------

Basic Client Usage
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from justllms import Client

   # Initialize with defaults
   client = Client()

   # Initialize with configuration file
   client = Client(config="config.yaml")

   # Initialize with dict configuration
   client = Client(config={
       "routing": {"strategy": "cost"},
       "cache": {"enabled": True}
   })

Creating Completions
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Simple completion
   response = client.completion.create(
       messages=[{"role": "user", "content": "Hello!"}],
       model="gpt-3.5-turbo"
   )

   # With additional parameters
   response = client.completion.create(
       messages=[
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": "Explain recursion"}
       ],
       model="gpt-4",
       temperature=0.7,
       max_tokens=500,
       top_p=0.9
   )

   # Streaming
   for chunk in client.completion.create(
       messages=[{"role": "user", "content": "Tell me a story"}],
       stream=True
   ):
       print(chunk.content, end="")

Managing Providers
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # List available providers
   providers = client.list_providers()
   print(f"Available providers: {providers}")

   # List models from all providers
   models = client.list_models()
   for provider, provider_models in models.items():
       print(f"{provider}: {list(provider_models.keys())}")

   # Add a custom provider
   from my_module import CustomProvider
   client.add_provider("custom", CustomProvider(config))