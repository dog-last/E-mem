# Model Compatibility and KV Cache Management

## Overview

KV cache files are model-specific and cannot be reused across different models. Each model has unique:
- Hidden dimensions
- Number of layers
- Attention head configurations

Attempting to load KV cache from a different model will cause dimension mismatch errors.

## Model ID Tracking

Starting from this version, the system tracks `model_id` in both:
1. **Cache files** (`kv_cache_*.pt`) - Stored in cache_state
2. **Metadata file** (`agents_metadata.json`) - Stored in agent metadata

### Multi-Model Support

The metadata file supports **multiple models simultaneously**:
- Each model's KV cache blocks are stored with their `model_id`
- When loading, only blocks matching the current model are loaded
- Other models' blocks are preserved but skipped
- This allows switching between models without losing cache data

### Metadata Structure

```json
[
  {
    "block_id": "uuid-1",
    "timestamp": "20231201_120000",
    "model_id": "Qwen/Qwen3-4B",
    "summary": "...",
    "is_active": false,
    "block_used": 12195,
    "chunk_number": 194
  },
  {
    "block_id": "uuid-2",
    "timestamp": "20231201_130000",
    "model_id": "Qwen/Qwen3-1.7B",
    "summary": "...",
    "is_active": false,
    "block_used": 8500,
    "chunk_number": 120
  }
]
```

## Validation

The system validates model compatibility at two points:

1. **During agent loading** (`MemoryHandler._load_existing_agents`)
   - Checks metadata `model_id` against current model
   - Skips incompatible agents with a warning

2. **During cache loading** (`MemoryAgent.__init__` and `_agent_generate`)
   - Validates cache file `model_id` against current model
   - Raises `ValueError` if mismatch detected

## Error Messages

### Model Mismatch Error

```
ValueError: Model mismatch: cache was created with 'Qwen/Qwen3-4B' but trying to load with 'Qwen/Qwen3-7B'. 
KV cache dimensions are incompatible between different models. Please use clean_cache_first=True or use the same model.
```

### Solution

1. **Use the same model** - Continue with the original model
2. **Clear cache** - Start fresh with new model:
   ```python
   chat_manager = ChatManager(
       model_id="new-model",
       clean_cache_first=True  # Clear old cache
   )
   ```

## Migration Guide

### For Existing Users

If you have existing cache files without `model_id` in metadata:

1. **Run migration script**:
   ```bash
   python scripts/migrate_metadata.py
   # Or specify custom path
   python scripts/migrate_metadata.py /path/to/kv_data
   ```

2. **Manual migration**:
   - The script reads cache files to extract `model_id`
   - Updates `agents_metadata.json` with model information
   - Preserves all existing data

### What Happens Without Migration

- Old metadata without `model_id` will still load
- System logs warning: "Skipping agent {id}: model mismatch"
- No errors, but agents won't be loaded

## Best Practices

1. **Multi-Model Workflow**
   - You can safely switch between models without `clean_cache_first=True`
   - Each model maintains its own KV cache blocks
   - Metadata file stores all models' blocks together

2. **Clean Start When Needed**
   ```python
   # Only use clean_cache_first=True when you want to delete ALL cache
   chat_manager = ChatManager(
       model_id="any-model",
       clean_cache_first=True  # Deletes cache for ALL models
   )
   ```

3. **Model Switching**
   ```python
   # First session with 4B model
   chat_4b = ChatManager(model_id="Qwen/Qwen3-4B", clean_cache_first=False)
   # ... use it ...
   
   # Later session with 1.7B model (4B cache preserved)
   chat_1_7b = ChatManager(model_id="Qwen/Qwen3-1.7B", clean_cache_first=False)
   # ... use it ...
   
   # Back to 4B model (loads previous 4B cache)
   chat_4b_again = ChatManager(model_id="Qwen/Qwen3-4B", clean_cache_first=False)
   ```

4. **Check Logs**
   - Monitor info about loaded agents per model
   - Verify model_id in metadata after saving

## Technical Details

### Cache State Structure

```python
cache_state = {
    "global_offset": int,
    "saved_chunks": List[Dict],
    "chunk_number": int,
    "model_id": str,  # Added for compatibility checking
    "merged_cache": List[Tuple[Tensor, Tensor]]
}
```

### Validation Logic

```python
# In MemoryAgent.__init__
cached_model_id = cache_state.get("model_id")
if cached_model_id and cached_model_id != model_id:
    raise ValueError(f"Model mismatch: ...")

# In MemoryHandler._load_existing_agents
cached_model_id = agent_data.get("model_id")
if cached_model_id and cached_model_id != self.model_id:
    logger.warning(f"Skipping agent: model mismatch")
    continue
```

## FAQ

**Q: Can I use quantized and non-quantized versions of the same model?**

A: No. Even with the same base model, quantization changes the internal representations. Use `clean_cache_first=True` when switching.

**Q: What if I have multiple model sizes (e.g., 4B and 7B)?**

A: Each model automatically maintains its own cache blocks in the same metadata file. You can switch between models freely without clearing cache.

**Q: Will this affect my existing cache?**

A: No. Old cache files remain valid. Run the migration script to add model_id to metadata for better compatibility checking.

**Q: Can I manually edit metadata?**

A: Yes, but ensure `model_id` matches the cache file. Use the migration script for safety.
