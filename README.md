# KV-Cached Memory Agent System

A novel approach to LLM memory management using KV cache for efficient context handling in conversational AI.

## 🎯 Core Concept

Instead of traditional RAG-based memory retrieval, this system:
- **Stores memories as KV cache** - Faster prefilling compared to text re-encoding
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
pip install -r requirements.txt
```

## 📖 Usage

### Basic Example

```python
from src.conversation_manager.chat_handler import ChatManager

# Initialize
chat_manager = ChatManager(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    openai_config={"api_key": "your-key"},
    clean_cache_first=True
)

# Chat with memory
response = chat_manager.chat(
    user_input="My favorite color is blue.",
    auto_save=False  # Agent decides when to save
)
```

### Run Interactive Demo

```bash
python main.py
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
3. **Efficient Prefilling** - KV cache reuse speeds up context loading
4. **Scalable** - Parallel agents handle growing memory
5. **Minimal Components** - Pure LLM-based, no external vector DBs

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
├── src/
│   ├── agent/
│   │   └── base.py              # Base agent with tool calling
│   ├── conversation_manager/
│   │   ├── chat_handler.py      # User interface
│   │   └── user_loop.py         # (Optional) CLI loop
│   ├── memory/
│   │   ├── core/
│   │   │   └── loop_handler.py  # Memory orchestration
│   │   ├── kv_block_manager/
│   │   │   └── block.py         # KV cache storage
│   │   ├── memory_agent/
│   │   │   └── agent.py         # Memory agent logic
│   │   └── router/
│   │       └── router.py        # LLM-based router
│   └── utils/
│       └── prompt.py            # System prompts
├── main.py                      # Entry point
├── requirements.txt
└── README.md
```

## 🔬 Technical Details

### KV Cache Mechanism
- Each memory chunk is processed through the model once
- Only NEW cache is stored per chunk (not merged cache)
- During query, all chunks are merged and reused
- Position IDs ensure correct RoPE embeddings

### Memory Lifecycle
1. User provides information
2. Active agent adds to KV cache
3. When block full (90% of context window):
   - Agent generates summary
   - Moves to inactive pool
   - New active agent created
4. Queries hit both active (recent) and inactive (historical) agents

## 📝 License

See LICENSE file for details.
