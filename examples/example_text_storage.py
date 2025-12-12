"""Example demonstrating text storage mode vs KV cache mode."""
import logging

from src.conversation_manager.factory import create_chat_manager

logging.basicConfig(level=logging.INFO)


def main():
    # Example 1: Using KV Cache mode (default)
    print("=" * 50)
    print("Example 1: KV Cache Mode")
    print("=" * 50)
    
    kv_chat_manager = create_chat_manager(
        storage_mode="kv_cache",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        openai_config={"api_key": "your-openai-key", "base_url": "your-base-url"},
        clean_cache_first=True
    )
    
    # Add memory
    response = kv_chat_manager.chat(
        user_input="My favorite color is blue.",
        auto_save=True
    )
    print(f"KV Cache Response: {response}")
    
    # Query memory
    response = kv_chat_manager.chat(
        user_input="What is my favorite color?",
        auto_save=False
    )
    print(f"KV Cache Query: {response}\n")
    
    
    # Example 2: Using Text Storage mode
    print("=" * 50)
    print("Example 2: Text Storage Mode")
    print("=" * 50)
    
    text_chat_manager = create_chat_manager(
        storage_mode="text",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        openai_config={"api_key": "your-openai-key", "base_url": "your-base-url"},
        clean_cache_first=True
    )
    
    # Add memory
    response = text_chat_manager.chat(
        user_input="My favorite color is blue.",
        auto_save=True
    )
    print(f"Text Storage Response: {response}")
    
    # Query memory
    response = text_chat_manager.chat(
        user_input="What is my favorite color?",
        auto_save=False
    )
    print(f"Text Storage Query: {response}\n")


if __name__ == "__main__":
    main()
