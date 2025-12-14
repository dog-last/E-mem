"""Example demonstrating text storage mode."""
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.conversation_manager.factory import create_chat_manager

# Load config
try:
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    MODEL_ID = config['model']['model_id']
    OPENAI_CONFIG = config['model']['openai_config']
    MODEL_CONTEXT_WINDOW = config['model']['model_context_window']
    ROUTER_SYSTEM_PROMPT = config['memory'].get('router_system_prompt')
except Exception as e:
    print(f"Error loading config: {e}")
    exit(1)


def main():
    print("=" * 60)
    print("Text Storage Mode Example")
    print("=" * 60)
    
    # Create text storage manager
    OVERLAP_MODE = config['memory'].get('overlap_mode', 'chunk')
    manager = create_chat_manager(
        storage_mode="text",
        model_id=MODEL_ID,
        openai_config=OPENAI_CONFIG,
        model_context_window=MODEL_CONTEXT_WINDOW,
        router_system_prompt=ROUTER_SYSTEM_PROMPT,
        clean_cache_first=True,
        overlap_mode=OVERLAP_MODE
    )
    
    print("\n1. Adding memories...")
    memories = [
        "My name is Alice and I work at Google.",
        "I love programming in Python.",
        "My favorite hobby is hiking."
    ]
    
    for memory in memories:
        response = manager.chat(memory, auto_save=True)
        print(f"   Added: {memory}")
    
    print("\n2. Querying memories...")
    queries = [
        "What is my name?",
        "Where do I work?",
        "What do I like to do?"
    ]
    
    for query in queries:
        print(f"\n   Q: {query}")
        response = manager.chat(query, auto_save=False)
        print(f"   A: {response}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
