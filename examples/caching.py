import time
from justllms import JustLLM

# Client with caching enabled
client = JustLLM({
    "providers": {
        "google": {"api_key": "YOUR_GOOGLE_API_KEY"},
    },
    "cache": {
        "enabled": True,
        "ttl": 300  # 5 minutes
    }
})

prompt = "What is machine learning?"

# First request (no cache)
print("First request (creates cache):")
start = time.time()
response1 = client.completion.create(
    messages=[{"role": "user", "content": prompt}]
)
time1 = time.time() - start
print(f"Time: {time1:.2f} seconds")
print(f"Response: {response1.content[:100]}...\n")

# Second request (uses cache)
print("Second request (uses cache):")
start = time.time()
response2 = client.completion.create(
    messages=[{"role": "user", "content": prompt}]
)
time2 = time.time() - start
print(f"Time: {time2:.2f} seconds")
print(f"Response: {response2.content[:100]}...")

speedup = time1 / time2 if time2 > 0 else float('inf')
print(f"\n⚡ Speedup: {speedup:.1f}x faster")

# Different request (no cache match)
print("\nDifferent request (no cache):")
start = time.time()
response3 = client.completion.create(
    messages=[{"role": "user", "content": "What is deep learning?"}]
)
time3 = time.time() - start
print(f"Time: {time3:.2f} seconds")
print(f"Response: {response3.content[:100]}...")