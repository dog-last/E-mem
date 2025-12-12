"""Example comparing KV Cache and Text Storage modes."""
import logging

from src.conversation_manager.factory import create_chat_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_kv_cache_mode():
    """Demonstrate KV Cache storage mode."""
    print("\n" + "="*60)
    print("KV CACHE MODE DEMO")
    print("="*60)
    
    chat_manager = create_chat_manager(
        storage_mode="kv_cache",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        openai_config={"api_key": "your-openai-key", "base_url": "your-base-url"},
        clean_cache_first=True
    )
    
    # Add memory
    response = chat_manager.chat("My favorite color is blue.", auto_save=True)
    print(f"Add memory response: {response}")
    
    # Query memory
    response = chat_manager.chat("What is my favorite color?")
    print(f"Query response: {response}")


def demo_text_storage_mode():
    """Demonstrate Text Storage mode."""
    print("\n" + "="*60)
    print("TEXT STORAGE MODE DEMO")
    print("="*60)
    
    chat_manager = create_chat_manager(
        storage_mode="text",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        openai_config={"api_key": "your-openai-key", "base_url": "your-base-url"},
        clean_cache_first=True
    )
    
    # Add memory
    response = chat_manager.chat("My favorite color is blue.", auto_save=True)
    print(f"Add memory response: {response}")
    
    # Query memory
    response = chat_manager.chat("What is my favorite color?")
    print(f"Query response: {response}")


if __name__ == "__main__":
    print("Storage Mode Comparison Demo")
    print("This example shows how to use both KV Cache and Text Storage modes.\n")
    
    # Uncomment the mode you want to test:
    # demo_kv_cache_mode()
    # demo_text_storage_mode()
    
    print("\nNote: Update openai_config with your actual API credentials before running.")
