from justllms import JustLLM

client = JustLLM({
    "providers": {
        "google": {"api_key": "YOUR_GOOGLE_API_KEY"},
    }
})

# Basic streaming
print("🤖 Assistant: ", end="", flush=True)
stream = client.completion.create(
    messages=[{"role": "user", "content": "Write a short story about AI"}],
    stream=True
)

for chunk in stream:
    if chunk.content:
        print(chunk.content, end="", flush=True)
print("\n")

# Async streaming
import asyncio

async def async_streaming():
    print("\n🤖 Async Assistant: ", end="", flush=True)
    
    stream = client.completion.acreate(
        messages=[{"role": "user", "content": "Explain Python in simple terms"}],
        stream=True
    )
    
    async for chunk in stream:
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")

# Run async example
asyncio.run(async_streaming())