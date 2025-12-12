# Text Storage Mode Documentation

## Quick Links

- **[QUICKSTART_TEXT_STORAGE.md](QUICKSTART_TEXT_STORAGE.md)** - 30-second quick start
- **[TEXT_STORAGE_README.md](TEXT_STORAGE_README.md)** - Detailed documentation
- **[ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md)** - KV Cache vs Text Storage
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Implementation details

## Quick Start

```python
from src.conversation_manager.factory import create_chat_manager

# Text Storage mode (no GPU required)
manager = create_chat_manager(
    storage_mode="text",
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    openai_config={"api_key": "your-key"}
)

# Use it
manager.chat("My name is Alice.", auto_save=True)
response = manager.chat("What is my name?")
```

## When to Use

- ✅ No GPU available
- ✅ Using cloud LLM APIs
- ✅ Easy debugging needed
- ✅ Simple deployment

## Examples

See [../examples/example_text_storage.py](../examples/example_text_storage.py)

## Tests

Run tests:
```bash
python3 tests/test_text_storage.py
python3 tests/verify_implementation.py
```
