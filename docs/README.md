# Documentation

Complete documentation for the KV-Cached Memory Agent System.

## Quick Navigation

| Document | Description |
|----------|-------------|
| [API_REFERENCE.md](API_REFERENCE.md) | Complete API documentation |
| [QUICKSTART_TEXT_STORAGE.md](QUICKSTART_TEXT_STORAGE.md) | Text storage quick start guide |
| [ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md) | KV Cache vs Text Storage comparison |
| [PERSISTENCE.md](PERSISTENCE.md) | KV cache persistence details |
| [MODEL_COMPATIBILITY.md](MODEL_COMPATIBILITY.md) | Model compatibility guide |

---

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv sync

# Using pip
pip install -r requirements.txt

# Setup config
cp config.example.yaml config.yaml
```

### Basic Usage

```python
from src.conversation_manager import create_chat_manager

# Create manager (auto-selects based on storage_mode)
manager = create_chat_manager(
    storage_mode="kv_cache",  # or "text"
    model_id="Qwen/Qwen3-4B",
    openai_config={
        "api_key": "your-key",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini"
    }
)

# Store information
manager.chat("My name is Alice.", auto_save=True)

# Query information
response = manager.chat("What is my name?")
print(response)
```

---

## Storage Modes

### KV Cache Mode (Default)

- **Requires**: GPU with sufficient VRAM
- **Storage**: Binary `.pt` files
- **Best for**: Local deployment with GPU

```python
manager = create_chat_manager(
    storage_mode="kv_cache",
    model_id="Qwen/Qwen3-4B",
    openai_config={"api_key": "your-key"},
    max_memory={"0": "20GB"}  # GPU memory allocation
)
```

### Text Storage Mode

- **Requires**: OpenAI-compatible API access
- **Storage**: Human-readable JSON files
- **Best for**: Cloud deployment, no GPU

```python
manager = create_chat_manager(
    storage_mode="text",
    model_id="Qwen/Qwen3-4B",
    openai_config={"api_key": "your-key"}
)
```

---

## Configuration

### Using YAML Configuration

```yaml
# config.yaml
model:
  model_id: "Qwen/Qwen3-4B"
  openai_config:
    api_key: "your-api-key"
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o-mini"
  model_context_window: 32768

memory:
  storage_mode: "kv_cache"
  clean_cache_first: true
  overlap_ratio: 0.1
  overlap_mode: "chunk"
  block_size_ratio: 0.125
```

### Loading with Validation

```python
from config import load_validated_config

config = load_validated_config("config.yaml")
manager = create_chat_manager(
    storage_mode=config.memory.storage_mode,
    model_id=config.model.model_id,
    openai_config=config.model.openai_config.model_dump()
)
```

---

## Memory Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `overlap_ratio` | 0.0-0.5 | 0.1 | Overlap between memory blocks |
| `overlap_mode` | chunk/token | chunk | How overlap is calculated |
| `block_size_ratio` | 0.0-1.0 | 0.125 | Block size vs context window |
| `max_concurrent_gpu_operations` | 1-8 | 2 | Parallel GPU operations |
| `max_memory_segments` | 1-∞ | None | Max segments returned per query (None=unlimited) |
| `max_blocks` | 1-∞ | 5 | Max memory blocks selected by router |

### Query Result Limiting

The `max_memory_segments` parameter controls how many `<memory_segment>` elements are returned in query responses. This is useful for:

- **Reducing response length** when dealing with large memory blocks
- **Controlling LLM context** in downstream aggregation
- **Improving response relevance** by keeping only top segments

```python
# Limit to top 3 segments per query
manager = create_chat_manager(
    storage_mode="kv_cache",
    max_memory_segments=3,
    max_blocks=5  # Also limit router block selection
)
```

### Overlap Modes

- **chunk**: Keeps complete chunks in overlap buffer
- **token**: Accumulates sentences up to token limit

---

## Project Structure

```
mem-with-kv-cache/
├── src/
│   ├── agent/                    # Base agent with tool calling
│   ├── config/                   # Pydantic configuration schemas
│   │   ├── __init__.py
│   │   └── schema.py             # Configuration models
│   ├── conversation_manager/
│   │   ├── base_chat_manager.py  # Shared base class
│   │   ├── chat_handler.py       # ChatManager implementations
│   │   └── factory.py            # create_chat_manager()
│   ├── memory/
│   │   ├── core/                 # MemoryHandler, TextMemoryHandler
│   │   ├── kv_block_manager/     # KVBlock storage
│   │   ├── memory_agent/         # MemoryAgent, TextMemoryAgent
│   │   └── router/               # LLM-based routing
│   └── utils/                    # Prompts and utilities
├── evaluation/
│   ├── locomo/                   # LoComo benchmark
│   └── hotpotqa/                 # HotpotQA benchmark
├── tests/                        # 145 unit tests
├── docs/                         # This documentation
├── config.example.yaml           # Configuration template
└── config.py                     # Configuration loader
```

---

## Running Examples

```bash
# Quick start example
python examples/quickstart.py

# Text storage mode
python examples/example_text_storage.py

# Simple example
python examples/example_simple.py
```

---

## Running Evaluations

### LoComo Evaluation

```bash
# Basic run
python evaluation/locomo/eval_locomo.py

# With options
python evaluation/locomo/eval_locomo.py \
    --config config.yaml \
    --ratio 0.1 \
    --conversation_auto_save
```

### HotpotQA Evaluation

```bash
python evaluation/hotpotqa/eval_hotpotqa.py \
    --config evaluation/hotpotqa/config.yaml \
    --start-idx 0 \
    --end-idx 10
```

---

## Running Tests

```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test file
pytest tests/test_chat_manager.py -v

# Run pre-commit checks
pre-commit run --all-files
```

---

## Import Reference

```python
# Main interface
from src.conversation_manager import create_chat_manager
from src.conversation_manager import ChatManager, TextStorageChatManager

# Configuration
from src.config import AppConfig, MemoryConfig, OpenAIConfig
from config import load_validated_config, ConfigurationError

# Memory handlers
from src.memory.core.loop_handler import MemoryHandler
from src.memory.core.text_loop_handler import TextMemoryHandler

# Evaluation
from evaluation.locomo.load_dataset import load_locomo_dataset
from evaluation.locomo.utils import calculate_metrics
```

---

## Troubleshooting

### Config not found

```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your settings
```

### Import errors

Ensure you're running from the project root directory.

### GPU memory issues

1. Use text storage mode: `storage_mode="text"`
2. Reduce `model_context_window`
3. Enable quantization in `quantization_config`
4. Adjust `max_memory` per GPU

### Configuration validation errors

```python
from config import load_validated_config, ConfigurationError

try:
    config = load_validated_config("config.yaml")
except ConfigurationError as e:
    print(f"Fix these configuration issues:\n{e}")
```
