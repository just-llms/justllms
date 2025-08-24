from justllms import JustLLM

# Cost-optimized routing
client = JustLLM(
    {
        "providers": {
            "google": {"api_key": "YOUR_GOOGLE_API_KEY"},
        },
        "routing": {"strategy": "least_cost"},
    }
)

# Automatic routing (no model specified)
response = client.completion.create(messages=[{"role": "user", "content": "What is AI?"}])
print("Auto-routed to:", response.model)
print("Response:", response.content)

# Manual model selection
response = client.completion.create(
    messages=[{"role": "user", "content": "Explain neural networks"}],
    model="google/gemini-2.5-flash",
)
print(f"\nManual selection: {response.model}")
print("Response:", response.content)

# Performance-based routing
perf_client = JustLLM(
    {
        "providers": {
            "google": {"api_key": "YOUR_GOOGLE_API_KEY"},
        },
        "routing": {"strategy": "fastest"},
    }
)

response = perf_client.completion.create(
    messages=[{"role": "user", "content": "List 5 programming languages"}]
)
print(f"\nFastest model: {response.model}")
print("Response:", response.content)
