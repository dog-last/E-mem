"""
Simple example demonstrating the KV-cached memory system.
This example shows basic memory addition and querying without the full chat interface.
"""

from src.memory.core.loop_handler import MemoryHandler


def main():
    print("=" * 60)
    print("Simple Memory System Example")
    print("=" * 60)
    
    # Configuration
    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # Change to your local model path
    OPENAI_CONFIG = {
        "api_key": "your-openai-api-key",
        "base_url": "https://api.openai.com/v1"
    }
    
    print("\n1. Initializing memory handler...")
    memory_handler = MemoryHandler(
        model_id=MODEL_ID,
        openai_config=OPENAI_CONFIG,
        clean_cache_first=True,
        model_context_window=32768
    )
    print("   Memory handler initialized!")
    
    # Add some memories
    print("\n2. Adding memories...")
    memories = [
        "My name is Alice and I love programming in Python.",
        "I work as a machine learning engineer at TechCorp.",
        "My favorite hobby is hiking in the mountains on weekends.",
        "I have a pet cat named Whiskers who is 3 years old."
    ]
    
    for i, memory in enumerate(memories, 1):
        print(f"   Adding memory {i}: {memory[:50]}...")
        memory_handler.add_memory(memory)
    
    print("\n3. Querying memories...")
    
    # Test queries
    queries = [
        "What is my name?",
        "What do I do for work?",
        "Tell me about my hobbies.",
        "Do I have any pets?"
    ]
    
    for query in queries:
        print(f"\n   Q: {query}")
        response = memory_handler.query_memory(query)
        print(f"   A: {response}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
