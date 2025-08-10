from justllms import JustLLM

client = JustLLM({
    "providers": {
        "google": {"api_key": "YOUR_GOOGLE_API_KEY"},
    }
})

# Track costs across requests
total_cost = 0
requests = [
    "What is Python?",
    "What is JavaScript?", 
    "What is machine learning?"
]

for i, prompt in enumerate(requests, 1):
    response = client.completion.create(
        messages=[{"role": "user", "content": prompt}]
    )
    
    cost = response.usage.estimated_cost
    total_cost += cost
    
    print(f"Request {i}: ${cost:.6f}")
    print(f"Tokens: {response.usage.total_tokens}")
    print(f"Response: {response.content[:50]}...\n")

print(f"💰 Total Cost: ${total_cost:.6f}")

# Budget monitoring
budget = 0.05
spent = 0.0

print(f"\n📊 Budget: ${budget:.2f}")

for prompt in ["Explain AI", "What is ML?", "How does NLP work?"]:
    if spent >= budget:
        print("🚫 Budget exceeded!")
        break
        
    response = client.completion.create(
        messages=[{"role": "user", "content": prompt}]
    )
    
    cost = response.usage.estimated_cost
    spent += cost
    
    print(f"Spent: ${spent:.6f} / ${budget:.2f}")
    print(f"Remaining: ${budget - spent:.6f}")