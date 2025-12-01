"""
Main entry point for the KV-cached memory agent system.
This demonstrates a simple conversation loop with memory storage and retrieval.
"""

from src.conversation_manager.chat_handler import ChatManager

try:
    from config import (
        ATTN_IMPLEMENTATION,
        CLEAN_CACHE_ON_START,
        DEVICE_MAP,
        MAX_MEMORY,
        MAX_NEW_TOKENS,
        MODEL_CONTEXT_WINDOW,
        MODEL_ID,
        OFFLOAD_FOLDER,
        OPENAI_CONFIG,
        get_quantization_config,
    )
except ImportError:
    print("Warning: config.py not found. Using default configuration.")
    print("Copy config.example.py to config.py and customize it.\n")
    MODEL_ID = "Qwen/Qwen3-0.6B"
    OPENAI_CONFIG = {"api_key": "your-api-key-here"}
    MODEL_CONTEXT_WINDOW = 32768
    ATTN_IMPLEMENTATION = "sdpa"
    DEVICE_MAP = "auto"
    MAX_MEMORY = None
    OFFLOAD_FOLDER = None
    CLEAN_CACHE_ON_START = True
    MAX_NEW_TOKENS = 1024
    
    def get_quantization_config():
        return None


def main():
    
    print("=" * 60)
    print("KV-Cached Memory Agent System")
    print("=" * 60)
    print("\nInitializing chat manager...")
    
    # Initialize chat manager
    chat_manager = ChatManager(
        model_id=MODEL_ID,
        openai_config=OPENAI_CONFIG,
        clean_cache_first=CLEAN_CACHE_ON_START,
        model_context_window=MODEL_CONTEXT_WINDOW,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map=DEVICE_MAP,
        quantization_config=get_quantization_config(),
        max_memory=MAX_MEMORY,
        offload_folder=OFFLOAD_FOLDER
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
