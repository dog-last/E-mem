"""Example comparing KV Cache and Text Storage modes."""
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
    ATTN_IMPLEMENTATION = config['model']['attn_implementation']
    DEVICE_MAP = config['model']['device_map']
    QUANTIZATION_CONFIG = config['model'].get('quantization_config')
    MAX_MEMORY = config.get('max_memory')
    ROUTER_SYSTEM_PROMPT = config['memory'].get('router_system_prompt')
except Exception as e:
    print(f"Error loading config: {e}")
    exit(1)


def demo_mode(storage_mode: str):
    """Demo a specific storage mode."""
    print("\n" + "=" * 60)
    print(f"{storage_mode.upper().replace('_', ' ')} MODE")
    print("=" * 60)
    
    manager = create_chat_manager(
        storage_mode=storage_mode,
        model_id=MODEL_ID,
        openai_config=OPENAI_CONFIG,
        model_context_window=MODEL_CONTEXT_WINDOW,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map=DEVICE_MAP,
        router_system_prompt=ROUTER_SYSTEM_PROMPT,
        quantization_config=QUANTIZATION_CONFIG,
        max_memory=MAX_MEMORY,
        clean_cache_first=True
    )
    
    # Add memory
    print("\nAdding memory: 'My favorite color is blue.'")
    response = manager.chat("My favorite color is blue.", auto_save=True)
    print(f"Response: {response}")
    
    # Query memory
    print("\nQuerying: 'What is my favorite color?'")
    response = manager.chat("What is my favorite color?", auto_save=False)
    print(f"Response: {response}")


def main():
    print("=" * 60)
    print("Storage Mode Comparison")
    print("=" * 60)
    print("\nThis example demonstrates both storage modes using config.yaml")
    
    # Demo KV Cache mode
    demo_mode("kv_cache")
    
    # Demo Text Storage mode
    demo_mode("text")
    
    print("\n" + "=" * 60)
    print("Comparison completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
