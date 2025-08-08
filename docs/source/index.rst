.. JustLLMs documentation master file

Welcome to JustLLMs Documentation
==================================

JustLLMs is a unified Python gateway for multiple LLM providers with intelligent routing, monitoring, and caching.

.. image:: https://img.shields.io/pypi/v/justllms.svg
   :target: https://pypi.org/project/justllms/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/justllms.svg
   :target: https://pypi.org/project/justllms/
   :alt: Python versions

Features
--------

- 🔄 **Unified API Interface** - Single API format for 100+ LLMs
- 📊 **Intelligent Routing** - Cost, quality, and latency-based model selection
- 💰 **Cost Tracking** - Real-time usage monitoring and budget controls
- 🚀 **Performance** - Response caching, retries, and fallbacks
- 🔧 **Extensible** - Easy to add new providers and strategies

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install justllms

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from justllms import Client

   # Initialize client
   client = Client()

   # Create a completion
   response = client.completion.create(
       messages=[{"role": "user", "content": "Hello, how are you?"}],
       model="gpt-3.5-turbo"
   )

   print(response.content)

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   configuration
   providers
   routing
   monitoring
   caching
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/client
   api/providers
   api/routing
   api/monitoring
   api/cache
   api/exceptions

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   contributing
   extending
   testing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`