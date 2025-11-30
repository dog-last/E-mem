"""
Configuration example for the KV-cached memory system.
Copy this file to config.py and fill in your settings.
"""

# Model Configuration
MODEL_ID = "Qwen/Qwen3-0.6B"  # Path to your local model or HuggingFace model ID
MODEL_CONTEXT_WINDOW = 32768  # Context window size (block size will be 90% of this)
ATTN_IMPLEMENTATION = "sdpa"  # Attention implementation: "sdpa", "flash_attention_2", "eager"
DEVICE_MAP = "auto"  # Device mapping: "auto", "cuda", "cpu", or custom dict

# Quantization (optional)
USE_QUANTIZATION = False
QUANTIZATION_BITS = 4  # 4 or 8

# OpenAI Configuration (for Router agent)
OPENAI_CONFIG = {
    "api_key": "your-openai-api-key-here",
    "base_url": "https://api.openai.com/v1",  # Optional: for custom endpoints
    # "model": "gpt-4o-mini"  # Optional: specify model
}

# Memory Configuration
CLEAN_CACHE_ON_START = True  # Clear previous cache files on startup
MAX_TOOL_ROUNDS = 5  # Maximum rounds of tool calling

# Router Configuration
ROUTER_SYSTEM_PROMPT = None  # Use default if None

# Chat Configuration
CHAT_SYSTEM_PROMPT = None  # Use default if None
MAX_NEW_TOKENS = 1024  # Maximum tokens to generate in responses


def get_quantization_config():
    """Get quantization configuration if enabled."""
    if not USE_QUANTIZATION:
        return None
    
    import torch
    from transformers import BitsAndBytesConfig
    
    if QUANTIZATION_BITS == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif QUANTIZATION_BITS == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    else:
        raise ValueError(f"Unsupported quantization bits: {QUANTIZATION_BITS}")
