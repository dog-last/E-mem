# KV-Cached Memory Agent System

<p align="center">
  <strong>A novel approach to LLM memory management using KV cache for efficient context handling</strong>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-configuration">Configuration</a> •
  <a href="#-documentation">Documentation</a>
</p>

---

## 🌟 Features

- **Dual Storage Modes**: KV Cache (GPU) and Text Storage (API-based)
- **No Traditional RAG**: Avoids embedding-based search inaccuracies
- **Full Context Understanding**: LLM sees complete memory, not fragments
- **KV Cache Reuse**: Cached context for efficient memory access
- **Scalable Architecture**: Parallel agents handle growing memory
- **Pydantic Validation**: Type-safe configuration with schema validation
- **145 Unit Tests**: Comprehensive test coverage

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/mem-with-kv-cache.git
cd mem-with-kv-cache

# Install with uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt

# Setup configuration
cp config.example.yaml config.yaml
# Edit config.yaml with your settings
```

## 🚀 Quick Start

### Using the Factory (Recommended)

```python
from src.conversation_manager import create_chat_manager

# KV Cache mode (GPU required)
manager = create_chat_manager(
    storage_mode="kv_cache",
    model_id="Qwen/Qwen3-4B",
    openai_config={
        "api_key": "your-key",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini"
    }
)

# Store memory
manager.chat("My name is Alice and I love hiking.", auto_save=True)

# Query memory
response = manager.chat("What is my name and hobby?")
print(response)
```

### Text Storage Mode (No GPU Required)

```python
manager = create_chat_manager(
    storage_mode="text",
    model_id="Qwen/Qwen3-4B",
    openai_config={"api_key": "your-key", "model": "gpt-4o-mini"}
)
```

See [docs/QUICKSTART_TEXT_STORAGE.md](docs/QUICKSTART_TEXT_STORAGE.md) for detailed guide.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ChatManager (Interface)                   │
│                 ┌──────────────────────────┐                │
│                 │    BaseChatManager       │                │
│                 │  - Tool definitions      │                │
│                 │  - Memory aggregation    │                │
│                 └──────────────────────────┘                │
│                            │                                 │
│              ┌─────────────┴─────────────┐                  │
│              ▼                           ▼                  │
│    ┌─────────────────┐         ┌─────────────────┐         │
│    │   ChatManager   │         │ TextStorage     │         │
│    │   (KV Cache)    │         │ ChatManager     │         │
│    └────────┬────────┘         └────────┬────────┘         │
└─────────────┼───────────────────────────┼───────────────────┘
              │                           │
              ▼                           ▼
┌─────────────────────────┐    ┌─────────────────────────┐
│    MemoryHandler        │    │   TextMemoryHandler     │
│  ┌───────────────────┐  │    │  ┌───────────────────┐  │
│  │   AddHandler      │  │    │  │  TextAddHandler   │  │
│  │   (Active Agent)  │  │    │  │  (Text Storage)   │  │
│  └───────────────────┘  │    │  └───────────────────┘  │
│  ┌───────────────────┐  │    │  ┌───────────────────┐  │
│  │   QueryHandler    │  │    │  │ TextQueryHandler  │  │
│  │   (Router)        │  │    │  │   (Router)        │  │
│  └───────────────────┘  │    │  └───────────────────┘  │
└─────────────────────────┘    └─────────────────────────┘
              │                           │
              ▼                           ▼
       ┌──────────┐                ┌──────────┐
       │ KVBlock  │                │TextBlock │
       │ (.pt)    │                │ (.json)  │
       └──────────┘                └──────────┘
```

### Core Components

| Component | Description |
|-----------|-------------|
| **BaseChatManager** | Abstract base with shared tool definitions and memory aggregation |
| **ChatManager** | KV cache implementation for GPU-based storage |
| **TextStorageChatManager** | Text-based implementation for API-only deployment |
| **MemoryHandler** | Orchestrates memory addition and retrieval |
| **Router** | LLM-based intelligent routing using summaries |
| **KVBlock/TextBlock** | Storage backends for memory persistence |

## ⚙️ Configuration

### Configuration Schema (Pydantic Validated)

```yaml
# config.yaml
model:
  model_id: "Qwen/Qwen3-4B"
  openai_config:
    api_key: "your-api-key"
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o-mini"
  model_context_window: 32768
  attn_implementation: "sdpa"  # sdpa, flash_attention_2, eager
  device_map: "auto"

memory:
  storage_mode: "kv_cache"      # kv_cache or text
  clean_cache_first: true
  overlap_ratio: 0.1            # 0.0-0.5
  overlap_mode: "chunk"         # chunk or token
  block_size_ratio: 0.125       # 0.0-1.0
  max_concurrent_gpu_operations: 2
  max_memory_segments: 5        # Max segments returned per query (optional)
  max_blocks: 5                 # Max memory blocks selected by router

max_memory:
  0: "20GB"
  1: "20GB"
```

### Programmatic Configuration with Validation

```python
from src.config import load_and_validate_config, MemoryConfig

# Validate memory config
memory_config = MemoryConfig(
    storage_mode="kv_cache",
    overlap_ratio=0.1,
    block_size_ratio=0.125,
    max_memory_segments=5,
    max_blocks=5
)

# Load and validate full config
from config import load_validated_config
app_config = load_validated_config("config.yaml")
```

## 📁 Project Structure

```
mem-with-kv-cache/
├── src/
│   ├── agent/                    # Base agent with tool calling
│   ├── config/                   # Pydantic configuration schemas
│   ├── conversation_manager/     # Chat interface & factory
│   │   ├── base_chat_manager.py  # Shared base class
│   │   ├── chat_handler.py       # KV cache & text implementations
│   │   └── factory.py            # Factory function
│   ├── memory/
│   │   ├── core/                 # Memory handlers
│   │   ├── kv_block_manager/     # KV cache storage
│   │   ├── memory_agent/         # Memory agents
│   │   └── router/               # LLM-based routing
│   └── utils/                    # Utilities & prompts
├── evaluation/
│   ├── locomo/                   # LoComo benchmark
│   └── hotpotqa/                 # HotpotQA benchmark
├── tests/                        # 145 unit tests
├── docs/                         # Documentation
├── config.example.yaml           # Configuration template
└── config.py                     # Configuration loader
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_chat_manager.py -v
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [docs/README.md](docs/README.md) | Complete documentation index |
| [docs/QUICKSTART_TEXT_STORAGE.md](docs/QUICKSTART_TEXT_STORAGE.md) | Text storage quick start |
| [docs/ARCHITECTURE_COMPARISON.md](docs/ARCHITECTURE_COMPARISON.md) | KV Cache vs Text Storage |
| [docs/PERSISTENCE.md](docs/PERSISTENCE.md) | KV cache persistence |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | API reference |
| [evaluation/locomo/README.md](evaluation/locomo/README.md) | LoComo evaluation guide |

## 🎯 Use Cases

### ✅ Best For

- **Conversational AI**: Personal assistants, chatbots with memory
- **User-specific Data**: Individual user memory management
- **High Accuracy Needs**: Full context understanding required
- **Local Deployment**: GPU-based KV cache for efficiency

### ❌ Not Recommended For

- **Enterprise RAG**: Vector DBs more efficient at scale
- **Real-time Updates**: KV cache rebuild has overhead
- **Task Agents**: Traditional RAG better for tool-heavy workflows

## 🔬 Technical Details

### KV Cache Mechanism

1. Each memory chunk is processed through the model once
2. KV tensors are cached and stored to disk (`.pt` files)
3. During query, cached KV tensors are loaded and reused
4. Position IDs ensure correct RoPE embeddings

### Memory Lifecycle

```
User Input → Active Agent (add to KV cache)
                    ↓
            Block Full (90% capacity)?
                    ↓ Yes
            Generate Summary → Move to Inactive Pool
                    ↓
            Create New Active Agent
                    ↓
Query → Router (select by summary) → Parallel Query → Aggregate Results
```

## 📝 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

### Code Attribution

The evaluation code in `evaluation/hotpotqa/eval_hotpotqa.py` is adapted from the GAM framework (MIT License). See [NOTICE](NOTICE) for details.

## 🤝 Contributing

Contributions are welcome! Please ensure:

1. Run tests: `pytest tests/`
2. Run linting: `ruff check .`
3. Run pre-commit hooks: `pre-commit run --all-files`
