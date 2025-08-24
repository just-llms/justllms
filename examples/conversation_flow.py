from justllms import JustLLM
from justllms.conversations import Conversation

# Configure client
client = JustLLM({"providers": {"google": {"api_key": "YOUR_GOOGLE_API_KEY"}}})

# Create conversation with automatic context management
conversation = Conversation(client=client)

# Multi-turn conversation using Conversation.send()
print("🗣️  Multi-turn Conversation Example")
print("=" * 40)

# Set system message
conversation.add_system_message("You are a helpful math tutor. Keep answers concise.")

# Turn 1
response = conversation.send("What is 15 + 25?")
print("👤 User: What is 15 + 25?")
print(f"🤖 Assistant: {response.content}")

# Turn 2 - Context is automatically preserved
response = conversation.send("Now divide that by 8")
print("👤 User: Now divide that by 8")
print(f"🤖 Assistant: {response.content}")

# Turn 3 - Context continues automatically
response = conversation.send("What's the square root of that?")
print("👤 User: What's the square root of that?")
print(f"🤖 Assistant: {response.content}")

# Get conversation stats
history = conversation.get_history()
print("\n📊 Conversation Stats:")
print(f"Conversation ID: {conversation.id}")
print(f"Total messages: {len(history)}")
print(f"Last response tokens: {response.usage.total_tokens if response.usage else 'N/A'}")
print(
    f"Last response cost: ${response.usage.estimated_cost:.6f}" if response.usage else "Cost: N/A"
)
