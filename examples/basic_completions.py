from justllms import JustLLM

# Initialize client
client = JustLLM({
    "providers": {
        "google": {"api_key": "YOUR_GOOGLE_API_KEY"},
    }
})

# Basic completion
response = client.completion.create(
    messages=[{"role": "user", "content": "What is machine learning?"}]
)
print("Response:", response.content)
print("Model:", response.model)
print("Cost:", f"${response.usage.estimated_cost:.6f}")

# With system message
response = client.completion.create(
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python hello world program"}
    ]
)
print("\nCoding Assistant:", response.content)