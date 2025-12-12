# Quick Start: Text Storage Mode

## 30-Second Start

```python
from src.conversation_manager.factory import create_chat_manager

# Create Text Storage mode ChatManager
manager = create_chat_manager(
    storage_mode="text",
    model_id="Qwen/Qwen3-4B",
    openai_config={
        "api_key": "your-openai-api-key",
        "base_url": "https://api.openai.com/v1"
    }
)

# Add memory
manager.chat("My name is Alice and I love Python.", auto_save=True)

# Query memory
response = manager.chat("What is my name?")
print(response)
```

## Switch to KV Cache Mode

Just change one parameter:

```python
manager = create_chat_manager(
    storage_mode="kv_cache",  # Change to kv_cache
    model_id="Qwen/Qwen3-4B",
    openai_config={"api_key": "your-key"}
)
```

## Complete Example

```python
from src.conversation_manager.factory import create_chat_manager

# Initialize
manager = create_chat_manager(
    storage_mode="text",
    model_id="Qwen/Qwen3-4B",
    openai_config={"api_key": "sk-xxx"},
    clean_cache_first=True,
    model_context_window=32768
)

# Scenario 1: Auto-save mode
manager.chat("I work at Google as a software engineer.", auto_save=True)
manager.chat("My favorite programming language is Rust.", auto_save=True)

# Scenario 2: Let LLM decide whether to save
manager.chat("Remember that my birthday is on March 15th.")

# Scenario 3: Query memory
response = manager.chat("Where do I work?")
print(response)

response = manager.chat("What's my favorite language?")
print(response)
```

## Configuration

### Required Parameters
- `storage_mode`: `"text"` or `"kv_cache"`
- `model_id`: HuggingFace model ID (for tokenizer)
- `openai_config`: OpenAI API configuration

### Optional Parameters
- `clean_cache_first`: Clear cache on startup (default: True)
- `model_context_window`: Context window size (default: 32768)
- `router_system_prompt`: Custom router prompt

## Storage Location

- Text mode: `./text_data/*.json`
- KV Cache mode: `./kv_data/*.pt`

## Run Examples

```bash
# Run complete example
python examples/example_text_storage.py

# Run tests
pytest tests/test_text_storage.py
```

## FAQ

**Q: Can both modes be used simultaneously?**  
A: Yes, they use different storage directories and don't interfere with each other.

**Q: How to clear cache?**  
A: Set `clean_cache_first=True` or manually delete the `text_data/` directory.

**Q: Does Text mode require GPU?**  
A: No, only OpenAI API key is required.

**Q: What are the main differences?**  
A: KV Cache mode requires GPU and stores binary tensors. Text mode uses API calls and stores JSON files.

## Next Steps

- See [TEXT_STORAGE_README.md](TEXT_STORAGE_README.md) for detailed architecture
- See [ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md) for comparison
- Run `examples/example_text_storage.py` for complete example
