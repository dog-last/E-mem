"""
Main entry point for the KV-cached memory agent system.
This demonstrates a simple conversation loop with memory storage and retrieval.
"""

import os

import yaml

from src.conversation_manager.factory import create_chat_manager

# Try to load from config.yaml
try:
    config_path = os.path.join('evaluation', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    MODEL_ID = config['model']['model_id']
    OPENAI_CONFIG = config['model']['openai_config']
    MODEL_CONTEXT_WINDOW = config['model']['model_context_window']
    ATTN_IMPLEMENTATION = config['model']['attn_implementation']
    DEVICE_MAP = config['model']['device_map']
    STORAGE_MODE = config['memory'].get('storage_mode', 'kv_cache')
    CLEAN_CACHE_ON_START = config['memory']['clean_cache_first']
    MAX_NEW_TOKENS = 1024
    
    def get_quantization_config():
        return config['model'].get('quantization_config')
    
    def get_max_memory():
        return config.get('max_memory')
except Exception as e:
    print(f"Warning: Could not load config.yaml: {e}")
    print("Using default configuration.\n")
    MODEL_ID = "Qwen/Qwen3-0.6B"
    OPENAI_CONFIG = {"api_key": "your-api-key-here"}
    MODEL_CONTEXT_WINDOW = 32768
    ATTN_IMPLEMENTATION = "sdpa"
    DEVICE_MAP = "auto"
    STORAGE_MODE = "kv_cache"
    CLEAN_CACHE_ON_START = True
    MAX_NEW_TOKENS = 1024
    
    def get_quantization_config():
        return None
    
    def get_max_memory():
        return None


def main():
    
    print("=" * 60)
    print("KV-Cached Memory Agent System")
    print("=" * 60)
    print("\nInitializing chat manager...")
    
    # Initialize chat manager using factory
    chat_manager = create_chat_manager(
        storage_mode=STORAGE_MODE,
        model_id=MODEL_ID,
        openai_config=OPENAI_CONFIG,
        clean_cache_first=CLEAN_CACHE_ON_START,
        model_context_window=MODEL_CONTEXT_WINDOW,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map=DEVICE_MAP,
        quantization_config=get_quantization_config(),
        max_memory=get_max_memory()
    )
    
    print("Chat manager initialized successfully!")
    print("\nYou can now chat with the assistant.")
    print("The assistant will automatically store and retrieve memories.")
    print("Type 'quit' or 'exit' to end the conversation.\n")
    
    # Conversation loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye!")
                break
            
            # Chat with auto memory management
            print("\nAssistant: ", end="", flush=True)
            response = chat_manager.chat(
                user_input=user_input,
                auto_save=False,  # Let agent decide when to save
                save_original_input=False,  # Save processed memory, not raw input
                max_new_tokens=MAX_NEW_TOKENS
            )
            
            if response:
                print(response)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Continuing...")


if __name__ == "__main__":
    main()
