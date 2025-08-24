from justllms import JustLLM

# Basic configuration
client = JustLLM(
    {
        "providers": {
            "google": {"api_key": "YOUR_GOOGLE_API_KEY"},
        }
    }
)

response = client.completion.create(messages=[{"role": "user", "content": "Hello!"}])
print("Basic:", response.content)

# With custom timeout
client_timeout = JustLLM(
    {
        "providers": {
            "google": {
                "api_key": "YOUR_GOOGLE_API_KEY",
                "timeout": 10,  # 10 second timeout
            }
        }
    }
)

response = client_timeout.completion.create(
    messages=[{"role": "user", "content": "What is Python?"}]
)
print("Custom timeout:", response.content[:50], "...")

# With retry configuration
client_retry = JustLLM(
    {
        "providers": {
            "google": {"api_key": "YOUR_GOOGLE_API_KEY", "max_retries": 3, "retry_delay": 1.0}
        }
    }
)

response = client_retry.completion.create(messages=[{"role": "user", "content": "Explain AI"}])
print("With retries:", response.content[:50], "...")

# Multiple providers (if you had them)
# client_multi = JustLLM({
#     "providers": {
#         "google": {"api_key": "YOUR_GOOGLE_API_KEY"},
#         "openai": {"api_key": "YOUR_OPENAI_API_KEY"},
#     },
#     "routing": {
#         "strategy": "least_cost"
#     }
# })
