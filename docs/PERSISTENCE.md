# KV Cache Persistence

## Overview

KV Cache mode now supports **automatic persistence** of memory agents. When you restart the application, all previous memory blocks are automatically restored.

## How It Works

### Metadata Storage

Agent metadata is saved to `kv_data/agents_metadata.json`:

```json
[
  {
    "block_id": "uuid-1",
    "timestamp": "20240101_120000",
    "summary": "Summary of this memory block...",
    "is_active": false,
    "block_used": 10000,
    "chunk_number": 50
  },
  {
    "block_id": "uuid-2",
    "timestamp": "20240101_130000",
    "summary": null,
    "is_active": true,
    "block_used": 5000,
    "chunk_number": 25
  }
]
```

### Auto-Save

Metadata is automatically saved when:
- A new active agent is created
- An agent becomes inactive (block full)
- Summary is generated for inactive agents

### Auto-Load

Metadata is automatically loaded when:
- `MemoryHandler` is initialized with `clean_cache_first=False`
- Agents are restored from disk

## Usage

### Enable Persistence (Default)

```python
from src.conversation_manager.chat_handler import ChatManager

# First run - creates new agents
manager = ChatManager(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    openai_config={"api_key": "your-key"},
    clean_cache_first=False  # Enable persistence
)

manager.chat("My name is Alice.", auto_save=True)
```

### Restart and Restore

```python
# Second run - restores previous agents
manager = ChatManager(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    openai_config={"api_key": "your-key"},
    clean_cache_first=False  # Load existing agents
)

# Previous memories are available
response = manager.chat("What is my name?")
# Output: "Your name is Alice."
```

### Clear All Data

```python
# Start fresh
manager = ChatManager(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    openai_config={"api_key": "your-key"},
    clean_cache_first=True  # Clear everything
)
```

## What Gets Persisted

### Saved
- ✅ Block UUID
- ✅ Creation timestamp
- ✅ Summary (for inactive agents only)
- ✅ Active/inactive status
- ✅ Block usage (tokens used)
- ✅ Chunk number
- ✅ KV cache (in .pt files)
- ✅ Chunk metadata

### Not Saved
- ❌ Overlap buffer (recreated on load)
- ❌ GPU cache (loaded on demand)

## File Structure

```
kv_data/
├── agents_metadata.json          # Agent metadata
├── kv_cache_uuid1_timestamp.pt   # KV cache for agent 1
└── kv_cache_uuid2_timestamp.pt   # KV cache for agent 2
```

## Benefits

1. **No Data Loss** - Memories persist across restarts
2. **Fast Startup** - Only metadata loaded initially
3. **Lazy Loading** - KV cache loaded on first query
4. **Automatic** - No manual save/load needed

## Example: Long-Running Assistant

```python
# Day 1
manager = ChatManager(model_id="...", clean_cache_first=True)
manager.chat("I work at Google.", auto_save=True)
# App closes

# Day 2
manager = ChatManager(model_id="...", clean_cache_first=False)
response = manager.chat("Where do I work?")
# Output: "You work at Google."
```

## Testing

Run persistence tests:
```bash
python3 tests/test_persistence.py
```

## Notes

- Metadata file is human-readable JSON
- KV cache files are binary PyTorch tensors
- Both are stored in `kv_data/` directory
- Use `clean_cache_first=True` to start fresh
