# API Reference

This document provides detailed API documentation for the KV-Cached Memory Agent System.

## Table of Contents

- [Factory Function](#factory-function)
- [Chat Managers](#chat-managers)
- [Configuration](#configuration)
- [Memory Handlers](#memory-handlers)

---

## Factory Function

### `create_chat_manager`

Factory function to create a chat manager with the specified storage backend.

```python
from src.conversation_manager import create_chat_manager

manager = create_chat_manager(
    storage_mode: Literal["kv_cache", "text"] = "kv_cache",
    model_id: str = "Qwen/Qwen3-4B",
    openai_config: dict = None,
    clean_cache_first: bool = True,
    model_context_window: int = 32768,
    router_system_prompt: str = None,
    overlap_mode: str = "chunk",
    overlap_ratio: float = 0.1,
    block_size_ratio: float = 0.125,
    **kwargs
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `storage_mode` | `Literal["kv_cache", "text"]` | `"kv_cache"` | Storage backend |
| `model_id` | `str` | `"Qwen/Qwen3-4B"` | HuggingFace model ID |
| `openai_config` | `dict` | `None` | OpenAI API configuration |
| `clean_cache_first` | `bool` | `True` | Clear cache on init |
| `model_context_window` | `int` | `32768` | Context window size |
| `router_system_prompt` | `str` | `None` | Custom router prompt |
| `overlap_mode` | `str` | `"chunk"` | `"chunk"` or `"token"` |
| `overlap_ratio` | `float` | `0.1` | Overlap ratio (0.0-0.5) |
| `block_size_ratio` | `float` | `0.125` | Block size ratio (0.0-1.0) |

**KV Cache Mode Additional Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `attn_implementation` | `str` | `"sdpa"` | Attention implementation |
| `device_map` | `str` | `"auto"` | Device mapping strategy |
| `quantization_config` | `dict` | `None` | Quantization config |
| `max_memory` | `dict` | `None` | Max memory per GPU |
| `offload_folder` | `str` | `None` | Offload folder path |

**Returns:** `ChatManager` or `TextStorageChatManager`

**Example:**

```python
# KV Cache mode
kv_manager = create_chat_manager(
    storage_mode="kv_cache",
    model_id="Qwen/Qwen3-4B",
    openai_config={"api_key": "your-key", "model": "gpt-4o-mini"},
    max_memory={"0": "20GB", "1": "20GB"}
)

# Text Storage mode
text_manager = create_chat_manager(
    storage_mode="text",
    model_id="Qwen/Qwen3-4B",
    openai_config={"api_key": "your-key", "model": "gpt-4o-mini"}
)
```

---

## Chat Managers

### `BaseChatManager`

Abstract base class providing shared functionality.

**Class Constants:**

```python
ADD_MEMORY_TOOL: dict   # Tool definition for add_memory
SEARCH_MEMORY_TOOL: dict  # Tool definition for query_memory
```

**Methods:**

#### `chat()`

Main chat interface method.

```python
def chat(
    self,
    user_input: str,
    outer_tools: Optional[List[dict]] = None,
    auto_save: bool = False,
    save_original_input: bool = False,
    max_new_tokens: int = 1024
) -> str
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_input` | `str` | - | User's input message |
| `outer_tools` | `List[dict]` | `None` | Additional tools |
| `auto_save` | `bool` | `False` | Auto-save without LLM |
| `save_original_input` | `bool` | `False` | Save original vs extracted |
| `max_new_tokens` | `int` | `1024` | Max tokens to generate |

**Returns:** `str` - The response

**Example:**

```python
# Let LLM decide to store
response = manager.chat("My favorite color is blue.")

# Force auto-save
manager.chat("[2024-01-01] Meeting notes...", auto_save=True)

# Query memory
response = manager.chat("What is my favorite color?")
```

#### `add_memory()`

Directly add memory without chat interface.

```python
def add_memory(self, memory: str) -> str
```

**Returns:** Success/error message string

#### `search_memory()`

Directly search memory without chat interface.

```python
def search_memory(self, query: str) -> str
```

**Returns:** Aggregated search results

---

### `ChatManager`

KV cache implementation for GPU-based storage.

```python
from src.conversation_manager import ChatManager

manager = ChatManager(
    model_id="Qwen/Qwen3-4B",
    openai_config={"api_key": "your-key"},
    model_context_window=32768,
    attn_implementation="sdpa",
    device_map="auto",
    block_size_ratio=0.125,
    max_memory_segments=5,  # Limit segments returned per query
    max_blocks=5            # Limit blocks selected by router
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | `str` | required | HuggingFace model ID or local path |
| `openai_config` | `dict` | required | OpenAI API configuration |
| `model_context_window` | `int` | `32768` | Model context window size |
| `attn_implementation` | `str` | `"sdpa"` | Attention implementation |
| `device_map` | `str` | `"auto"` | Device mapping strategy |
| `block_size_ratio` | `float` | `0.125` | Block size ratio (0.0-1.0) |
| `overlap_ratio` | `float` | `0.1` | Overlap ratio (0.0-0.5) |
| `max_memory_segments` | `int` | `None` | Max segments per query (None=unlimited) |
| `max_blocks` | `int` | `5` | Max blocks selected by router |

**Properties:**

- `name: str` - Returns `"chat_manager"`
- `memory_handler: MemoryHandler` - The memory handler instance

---

### `TextStorageChatManager`

Text-based implementation for API-only deployment.

```python
from src.conversation_manager import TextStorageChatManager

manager = TextStorageChatManager(
    model_id="Qwen/Qwen3-4B",
    openai_config={"api_key": "your-key"},
    model_context_window=32768,
    block_size_ratio=0.125,
    max_memory_segments=5,
    max_blocks=5
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | `str` | required | HuggingFace model ID (for tokenization) |
| `openai_config` | `dict` | required | OpenAI API configuration |
| `model_context_window` | `int` | `32768` | Model context window size |
| `block_size_ratio` | `float` | `0.125` | Block size ratio (0.0-1.0) |
| `overlap_ratio` | `float` | `0.1` | Overlap ratio (0.0-0.5) |
| `max_memory_segments` | `int` | `None` | Max segments per query (None=unlimited) |
| `max_blocks` | `int` | `5` | Max blocks selected by router |

**Properties:**

- `name: str` - Returns `"text_chat_manager"`
- `memory_handler: TextMemoryHandler` - The memory handler instance

---

## Configuration

### Configuration Schema Classes

All configuration is validated using Pydantic models.

```python
from src.config import (
    AppConfig,
    ModelConfig,
    MemoryConfig,
    OpenAIConfig,
    LoggingConfig,
    LocomoEvalConfig,
    HotpotQAEvalConfig,
)
```

### `OpenAIConfig`

```python
class OpenAIConfig(BaseModel):
    api_key: str          # Required
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
```

### `MemoryConfig`

```python
class MemoryConfig(BaseModel):
    storage_mode: Literal["kv_cache", "text"] = "kv_cache"
    clean_cache_first: bool = True
    router_system_prompt: Optional[str] = None
    overlap_ratio: float = 0.1        # 0.0-0.5
    overlap_mode: Literal["chunk", "token"] = "chunk"
    block_size_ratio: float = 0.125   # 0.0-1.0
    max_concurrent_gpu_operations: int = 2
    max_memory_segments: int = 5
    max_blocks: int = 5
```

### Loading Configuration

```python
from config import load_validated_config, ConfigurationError

try:
    config = load_validated_config("config.yaml")
    print(f"Model: {config.model.model_id}")
    print(f"Storage mode: {config.memory.storage_mode}")
except ConfigurationError as e:
    print(f"Invalid configuration: {e}")
```

### Standalone Validation

```python
from src.config import validate_memory_config, validate_openai_config

# Validate memory config
memory_config = validate_memory_config({
    "storage_mode": "kv_cache",
    "overlap_ratio": 0.1
})

# Validate OpenAI config
openai_config = validate_openai_config({
    "api_key": "your-key",
    "model": "gpt-4o-mini"
})
```

---

## Memory Handlers

### `MemoryHandler`

Orchestrates memory operations for KV cache mode.

```python
from src.memory.core.loop_handler import MemoryHandler

handler = MemoryHandler(
    model_id="Qwen/Qwen3-4B",
    openai_config={"api_key": "your-key"},
    clean_cache_first=True,
    model_context_window=32768,
    overlap_ratio=0.1
)
```

**Methods:**

```python
# Add memory
handler.add_memory(text: str) -> None

# Query memory
result = handler.query_memory(user_query: str) -> str
```

### `TextMemoryHandler`

Orchestrates memory operations for text storage mode.

```python
from src.memory.core.text_loop_handler import TextMemoryHandler

handler = TextMemoryHandler(
    model_id="Qwen/Qwen3-4B",
    openai_config={"api_key": "your-key"},
    clean_cache_first=True,
    model_context_window=32768
)
```

---

## Error Handling

### `ConfigurationError`

Raised when configuration validation fails.

```python
from config import ConfigurationError

try:
    config = load_validated_config("invalid.yaml")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

---

## Import Paths Summary

```python
# Main interface
from src.conversation_manager import create_chat_manager
from src.conversation_manager import ChatManager, TextStorageChatManager

# Configuration
from src.config import AppConfig, MemoryConfig, load_and_validate_config
from config import load_validated_config, ConfigurationError

# Memory handlers (low-level)
from src.memory.core.loop_handler import MemoryHandler
from src.memory.core.text_loop_handler import TextMemoryHandler

# Evaluation utilities
from evaluation.locomo.load_dataset import load_locomo_dataset
from evaluation.locomo.utils import calculate_metrics
```

