Exceptions API Reference
========================

.. module:: justllms.exceptions

JustLLMs provides a hierarchy of exceptions to help you handle different error scenarios gracefully.

Exception Hierarchy
-------------------

.. code-block:: text

   JustLLMsError (base exception)
   ├── ProviderError
   │   ├── RateLimitError
   │   ├── TimeoutError
   │   └── AuthenticationError
   ├── RouteError
   ├── ValidationError
   └── ConfigurationError

Base Exception
--------------

.. autoclass:: JustLLMsError
   :members:
   :undoc-members:
   :show-inheritance:

Provider Exceptions
-------------------

.. autoclass:: ProviderError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: RateLimitError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: TimeoutError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AuthenticationError
   :members:
   :undoc-members:
   :show-inheritance:

Other Exceptions
----------------

.. autoclass:: RouteError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ValidationError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ConfigurationError
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Error Handling
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from justllms import Client, JustLLMsError

   client = Client()

   try:
       response = client.completion.create(
           messages=[{"role": "user", "content": "Hello"}],
           model="gpt-3.5-turbo"
       )
   except JustLLMsError as e:
       print(f"JustLLMs error: {e.message}")
       if e.code:
           print(f"Error code: {e.code}")
       if e.details:
           print(f"Details: {e.details}")

Specific Error Handling
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from justllms import (
       Client, 
       ValidationError, 
       RateLimitError,
       TimeoutError,
       AuthenticationError
   )

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
           time.sleep(e.retry_after)
           # Retry the request
   except TimeoutError as e:
       print(f"Request timed out after {e.timeout_seconds}s")
   except AuthenticationError as e:
       print(f"Authentication failed: {e.message}")
       print(f"Required auth method: {e.required_auth}")

Provider Error Details
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from justllms import Client, ProviderError

   try:
       response = client.completion.create(
           messages=[{"role": "user", "content": "Hello"}],
           model="gpt-3.5-turbo"
       )
   except ProviderError as e:
       print(f"Provider: {e.provider}")
       print(f"Status code: {e.status_code}")
       print(f"Response body: {e.response_body}")
       
       # Check if it's a specific type of provider error
       if isinstance(e, RateLimitError):
           # Handle rate limiting
           pass
       elif isinstance(e, TimeoutError):
           # Handle timeout
           pass

Best Practices
--------------

1. **Always catch specific exceptions first**:

   .. code-block:: python

      try:
          # Your code
      except RateLimitError as e:
          # Handle rate limits specifically
      except ProviderError as e:
          # Handle other provider errors
      except JustLLMsError as e:
          # Handle any other JustLLMs error

2. **Use exception details for debugging**:

   .. code-block:: python

      except JustLLMsError as e:
          logger.error(
              f"Error: {e.message}",
              extra={
                  "code": e.code,
                  "details": e.details,
                  "exception_type": type(e).__name__
              }
          )

3. **Implement retry logic for transient errors**:

   .. code-block:: python

      from justllms.utils.retry import exponential_backoff

      @exponential_backoff(
          exceptions=(RateLimitError, TimeoutError),
          max_attempts=3
      )
      def make_request():
          return client.completion.create(...)

4. **Validate inputs to avoid ValidationError**:

   .. code-block:: python

      from justllms.utils.validators import validate_messages

      try:
          messages = validate_messages(user_input)
          response = client.completion.create(messages=messages)
      except ValidationError as e:
          return {"error": f"Invalid input: {e.message}"}