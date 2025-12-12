# Text Storage Mode

## Overview

This project now supports **two storage backends**:

1. **KV Cache Mode** (default) - Stores memories as KV cache for faster prefilling
2. **Text Storage Mode** (new) - Stores memories as plain text in JSON files

Both modes share the same architecture: block-based storage, summaries, router-based retrieval, and active/inactive agent management.

## Architecture Comparison

| Component | KV Cache Mode | Text Storage Mode |
|-----------|---------------|-------------------|
| **Storage** | `.pt` files in `kv_data/` | `.json` files in `text_data/` |
| **Memory Agent** | `MemoryAgent` (GPU-based) | `TextMemoryAgent` (API-based) |
| **Block Manager** | `KVBlock` | `TextBlock` |
| **Handler** | `MemoryHandler` | `TextMemoryHandler` |
| **Query Method** | KV cache + model.generate() | Full text + LLM API |
| **Speed** | Faster (cached prefill) | Slower (full text re-encoding) |
| **Resource** | GPU memory | API calls |

## Usage

### Method 1: Factory Function (Recommended)

```python
from src.conversation_manager.factory import create_chat_manager

# KV Cache mode
kv_manager = create_chat_manager(
    storage_mode="kv_cache",
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    openai_config={"api_key": "your-key"},
    clean_cache_first=True
)

# Text Storage mode
text_manager = create_chat_manager(
    storage_mode="text",
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    openai_config={"api_key": "your-key"},
    clean_cache_first=True
)
```

### Method 2: Direct Instantiation

```python
from src.conversation_manager.chat_handler import ChatManager, TextStorageChatManager

# KV Cache mode
kv_manager = ChatManager(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    openai_config={"api_key": "your-key"},
    clean_cache_first=True
)

# Text Storage mode
text_manager = TextStorageChatManager(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    openai_config={"api_key": "your-key"},
    clean_cache_first=True
)
```

### Chat Interface (Same for Both Modes)

```python
# Add memory
response = manager.chat(
    user_input="My favorite color is blue.",
    auto_save=True
)

# Query memory
response = manager.chat(
    user_input="What is my favorite color?",
    auto_save=False
)
```

## File Structure

```
src/
├── conversation_manager/
│   ├── chat_handler.py          # ChatManager & TextStorageChatManager
│   └── factory.py               # create_chat_manager()
├── memory/
│   ├── core/
│   │   ├── loop_handler.py      # KV cache handlers
│   │   └── text_loop_handler.py # Text storage handlers
│   ├── kv_block_manager/
│   │   ├── block.py             # KVBlock
│   │   └── text_block.py        # TextBlock
│   ├── memory_agent/
│   │   ├── agent.py             # MemoryAgent (KV cache)
│   │   └── text_agent.py        # TextMemoryAgent (text)
│   └── router/
│       └── router.py            # Shared router for both modes
```

## Storage Directories

- **KV Cache**: `kv_data/` (created automatically)
- **Text Storage**: `text_data/` (created automatically)

Both directories are created in the current working directory.

## When to Use Each Mode

### Use KV Cache Mode When:
- You have GPU resources available
- Speed is critical (faster prefilling)
- Working with conversational AI requiring low latency
- Memory scope is limited (user-specific data)

### Use Text Storage Mode When:
- No GPU available or limited GPU memory
- Using cloud LLM APIs (OpenAI, Anthropic, etc.)
- Easier debugging (human-readable JSON files)
- Simpler deployment (no model loading required)
- Cost-effective for low-frequency queries

## Example

See `example_text_storage.py` for a complete working example.

## API Compatibility

Both modes expose identical interfaces:
- `chat(user_input, auto_save, save_original_input, max_new_tokens)`
- `add_memory(memory)`
- `search_memory(query)`

You can switch between modes without changing your application code.
