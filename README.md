# E-mem

> **An Engram-Centric Paradigm for Episodic LLM Agent Memory Management**

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-key-features">Key Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-documentation">Documentation</a> •
  <a href="#-evaluation">Evaluation</a>
</p>

---

## 🚀 Overview

E-mem provides a unified interface for managing long-term memory in LLM applications. It abstracts the complexity of storage, offering two distinct modes:

1.  **KV Cache Mode (Main)**: The core operating mode. 
2.  **Text Storage Mode (Debug/API)**: A lightweight fallback. It stores raw text and relies on the LLM provider's context window. Ideal for debugging, environments without GPUs, or when using cloud-based inference endpoints that outperform local hardware.

## 🌟 Key Features

*   **⚡ Optimized Context**: Reuses cached KV tensors to reduce re-computation of history (latency depends on local hardware; Text Mode may be faster for cloud APIs).
*   **🧠 Hybrid Routing**: Advanced retrieval using a weighted mix of **Global Alignment** (Summary), **Semantic Association** (Dense), and **Symbolic Trigger** (Keyword).
*   **💾 Dual Persistence**: Seamlessly switch between `.pt` tensor storage and `.json` text storage.
*   **🛡️ Production Ready**: Type-safe configuration (Pydantic), automatic model compatibility checks, and robust error handling.
*   **🔌 Model Agnostic**: Supports HuggingFace models (local) and OpenAI-compatible APIs.

## 📦 Installation

We recommend using `uv` for dependency management

```bash
# Clone the repository
git clone https://github.com/your-username/mem-with-kv-cache.git
cd mem-with-kv-cache

# Install with uv (Recommended)
uv sync

# Setup configuration
cp config.example.yaml config.yaml
# Edit config.yaml with your settings
```

## ⚡ Quick Start

Get up and running in seconds. This factory pattern handles all complexity.

```python
from src.conversation_manager import create_chat_manager

# 1. Initialize Manager (KV Cache Mode)
manager = create_chat_manager(
    storage_mode="kv_cache",           # Use "text" for API-only mode
    model_id="Qwen/Qwen3-4B",          # HuggingFace Model ID
    openai_config={                    # For the LLM generation (can be local or remote)
        "api_key": "your-key",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini"
    }
)

# 2. Add Memory
# The system automatically chunks, embeds, and indexes this.
manager.chat("My name is Alice and I am a software engineer.", auto_save=True)

# 3. Query Memory
# The Hybrid Router retrieves relevant context automatically.
response = manager.chat("What is my profession?")
print(response)
# Output: "You are a software engineer."
```

## 📚 Documentation

Detailed documentation for architects and engineers.

| Resource | Description |
|:---|:---|
| **[Architecture](docs/ARCHITECTURE.md)** | Deep dive into system design, components, and the Hybrid Router. |
| **[Operational Guides](docs/GUIDES.md)** | Guides on Persistence, Text Mode, and Model Compatibility. |
| **[API Reference](docs/API_REFERENCE.md)** | Complete API specification for Managers, Handlers, and Config. |

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
│   │   └── router/               # Hybrid and LLM-based routing
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

Ensure system stability with our comprehensive test suite.

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_chat_manager.py -v
```

## 📊 Evaluation (Reproducing Results)

To reproduce the results presented in our paper, we provide automated scripts for both **LoComo** (Long-Context Memory) and **HotpotQA** (Multi-hop QA) benchmarks.

### LoComo Benchmark
Evaluates long-term memory retention and retrieval accuracy.

```bash
# Run LoComo evaluation
bash scripts/eval_locomo.sh
```

### HotpotQA Benchmark
Evaluates multi-hop reasoning capabilities using retrieved memory blocks.

```bash
# Run HotpotQA evaluation
bash scripts/eval_hotpotqa.sh
```

> Detailed evaluation configurations can be found in [`evaluation/locomo/README.md`](evaluation/locomo/README.md) and [`evaluation/hotpotqa/README.md`](evaluation/hotpotqa/README.md).

## 🤝 Contributing

We welcome contributions to E-mem! Please follow this workflow to ensure quality:

1.  **Fork & Clone**: Fork the repository and clone it locally.
2.  **Install Dependencies**:
    ```bash
    uv sync  
    ```
3.  **Setup Pre-commit**:
    Ensure code quality hooks are active.
    ```bash
    pre-commit install
    ```
4.  **Run Tests**:
    Verify that your changes don't break existing functionality.
    ```bash
    pytest tests/
    ```
5.  **Update Documentation**:
    If you modify the API or features, please update the relevant `docs/*.md` files or README.md file.

## 📄 License & Attribution

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

### Acknowledgements
This project includes some evaluation code adapted from the **GAM (General Agentic Memory)** framework. We explicitly acknowledge and thank the original authors. Please see [NOTICE](NOTICE) for detailed attribution and license information regarding third-party code.
