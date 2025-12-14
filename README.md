# KV-Cached Memory Agent System

A novel approach to LLM memory management using KV cache for efficient context handling in conversational AI.

## 🆕 NEW: Text Storage Mode

Now supports **two storage backends**:
- **KV Cache Mode** (default) - GPU-based
- **Text Storage Mode** (new) - API-based, no GPU required

👉 See [docs/QUICKSTART_TEXT_STORAGE.md](docs/QUICKSTART_TEXT_STORAGE.md) for quick start guide

## 🎯 Core Concept

Instead of traditional RAG-based memory retrieval, this system:
- **Stores memories as KV cache** - Reuses cached context
- **Parallel memory agents** - Multiple agents handle different memory blocks
- **Router-based selection** - Smart routing to relevant memory blocks using summaries
- **Long/short-term memory** - Active agent for recent memories, inactive agents for historical data

## 🏗️ Architecture

```
ChatManager (User Interface)
    |
    v
MemoryHandler (Orchestrator)
    |
    +-- AddHandler (Manages active memory agent)
    |       |
    |       v
    |   MemoryAgent (Active) --> KVBlock (KV Cache Storage)
    |
    +-- QueryHandler (Manages inactive agents)
            |
            v
        Router (LLM-based selector)
            |
            v
        [MemoryAgent (Inactive), MemoryAgent (Inactive), ...]
```

### Components

1. **KVBlock** (`src/memory/kv_block_manager/block.py`)
   - Stores KV cache to disk in `kv_data/` directory
   - Tracks token usage and block capacity
   - Automatically created in current working directory

2. **MemoryAgent** (`src/memory/memory_agent/agent.py`)
   - Incrementally builds KV cache from text chunks
   - Generates responses using cached context
   - Creates summaries when block becomes full
   - Active: accepts new memories | Inactive: query-only

3. **Router** (`src/memory/router/router.py`)
   - LLM-based routing using memory summaries
   - Parallel query execution with ThreadPoolExecutor
   - Returns top-k relevant memory blocks

4. **MemoryHandler** (`src/memory/core/loop_handler.py`)
   - Coordinates memory addition and retrieval
   - Manages agent lifecycle (active → inactive)
   - Parallel queries to active + inactive agents

5. **ChatManager** (`src/conversation_manager/chat_handler.py`)
   - User-facing interface with tool calling
   - Tools: `add_memory`, `query_memory`
   - Automatic memory management

## 🚀 Installation

```bash
# Install dependencies with uv
uv sync

# Or with pip
pip install -r requirements.txt

# Setup configuration
cp config.example.yaml config.yaml
# Edit config.yaml with your settings
```

## 📖 Usage

### Basic Example (KV Cache Mode)

```python
from src.conversation_manager.chat_handler import ChatManager

# Initialize
chat_manager = ChatManager(
    model_id="Qwen/Qwen3-4B",
    openai_config={"api_key": "your-key"},
    clean_cache_first=True
)

# Chat with memory
response = chat_manager.chat(
    user_input="My favorite color is blue.",
    auto_save=False  # Agent decides when to save
)
```

### Using Factory (Recommended)

```python
from src.conversation_manager.factory import create_chat_manager

# KV Cache mode (GPU required)
kv_manager = create_chat_manager(
    storage_mode="kv_cache",
    model_id="Qwen/Qwen3-4B",
    openai_config={"api_key": "your-key"}
)

# Text Storage mode (No GPU required)
text_manager = create_chat_manager(
    storage_mode="text",
    model_id="Qwen/Qwen3-4B",
    openai_config={"api_key": "your-key"}
)
```

### Run Examples

```bash
python examples/quickstart.py              # Quick start
python examples/example_text_storage.py    # Text storage mode
```

## 🔧 Configuration

### Model Quantization

```python
from transformers import BitsAndBytesConfig
import torch

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

chat_manager = ChatManager(
    model_id="your-model",
    quantization_config=quant_config
)
```

### Memory Block Size

```python
chat_manager = ChatManager(
    model_id="your-model",
    model_context_window=32768,  # Block size = 90% of this
)
```

## 💡 Key Advantages

1. **No Traditional Retrieval** - Avoids embedding-based search inaccuracies
2. **Full Context Understanding** - LLM sees complete memory, not fragments
3. **KV Cache Reuse** - Cached context for efficient memory access (KV Cache mode)
4. **Scalable** - Parallel agents handle growing memory
5. **Minimal Components** - Pure LLM-based, no external vector DBs
6. **Flexible Deployment** - Choose between GPU (KV Cache) or API (Text Storage)

## 🎓 Use Cases

### ✅ Suitable For:
- **Conversational AI** - Personal assistants, chatbots
- **Limited Memory Scope** - User-specific data (not enterprise-scale RAG)
- **High Accuracy Needs** - Full context understanding required

### ❌ Not Suitable For:
- **Task Completion Agents** - Better with traditional RAG for tool-heavy workflows
- **Massive Knowledge Bases** - Vector DBs more efficient at scale
- **Real-time Updates** - KV cache rebuild has overhead

## 📁 Project Structure

```
mem-with-kv-cache/
├── src/                         # Core source code
│   ├── agent/                   # Base agent with tool calling
│   ├── conversation_manager/    # Chat interface & factory
│   ├── memory/                  # Memory management (KV cache & text storage)
│   └── utils/                   # Utilities & prompts
├── examples/                    # Example scripts
│   ├── quickstart.py            # Quick start example
│   ├── example_text_storage.py  # Text storage example
│   └── example_simple.py        # Simple example
├── evaluation/locomo/           # LoComo dataset evaluation
│   ├── eval_locomo.py           # Main evaluation script
│   ├── load_dataset.py          # Dataset loader
│   ├── utils.py                 # Evaluation metrics
│   └── eval_data/               # Dataset files
├── scripts/                     # Utility scripts
│   └── run_eval.sh              # Evaluation runner
├── tests/                       # Unit tests
├── docs/                        # Documentation
├── config.yaml                  # Main config (gitignored)
├── config.example.yaml          # Config template
└── config.py                    # Config loader
```

## 🔬 Technical Details

### KV Cache Mechanism
- Each memory chunk is processed through the model once
- Only NEW cache is stored per chunk (not merged cache)
- During query, all chunks are merged and reused
- Position IDs ensure correct RoPE embeddings

### Persistence (KV Cache Mode)
- Agent metadata saved to `kv_data/agents_metadata.json`
- Includes: block UUID, timestamp, summary, active status
- Auto-saves when agent becomes inactive
- Auto-loads on startup (if `clean_cache_first=False`)
- See [docs/PERSISTENCE.md](docs/PERSISTENCE.md) for details

### Memory Lifecycle
1. User provides information
2. Active agent adds to KV cache
3. When block full (90% of context window):
   - Agent generates summary
   - Moves to inactive pool
   - New active agent created
4. Queries hit both active (recent) and inactive (historical) agents

## 📚 Documentation

- [docs/README.md](docs/README.md) - Complete documentation
- [docs/QUICKSTART_TEXT_STORAGE.md](docs/QUICKSTART_TEXT_STORAGE.md) - Text storage quick start
- [docs/ARCHITECTURE_COMPARISON.md](docs/ARCHITECTURE_COMPARISON.md) - KV Cache vs Text Storage
- [docs/PERSISTENCE.md](docs/PERSISTENCE.md) - KV cache persistence
- [docs/MODEL_COMPATIBILITY.md](docs/MODEL_COMPATIBILITY.md) - Model compatibility and migration
- [evaluation/locomo/README.md](evaluation/locomo/README.md) - Evaluation guide

## 🧪 Testing

```bash
pytest tests/                      # All tests
pytest tests/test_memory_agent.py  # Specific test
```

## 📂 Import Paths

```python
# Core components
from src.conversation_manager.factory import create_chat_manager
from src.conversation_manager.chat_handler import ChatManager

# Evaluation
from evaluation.locomo.load_dataset import load_locomo_dataset
from evaluation.locomo.utils import calculate_metrics

# Configuration
from config import MAX_CONCURRENT_GPU_OPERATIONS, DEFAULT_OVERLAP_RATIO
```

## 📝 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) file for details.

### Code Attribution

The evaluation code in `evaluation/hotpotqa/eval_hotpotqa.py` is adapted from the GAM (Generalized Augmented Memory) framework, which is licensed under the MIT License. See the [NOTICE](NOTICE) file for more information about the original source and license terms.