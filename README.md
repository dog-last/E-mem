# E-mem

> **Multi-agent based Episodic Context Reconstruction for LLM Agent Memory**

<p align="center">
  <a href="#-abstract">Abstract</a> •
  <a href="#-key-features">Key Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-documentation">Documentation</a> •
  <a href="#-evaluation">Evaluation</a>
</p>

---

## 📖 Abstract

**E-mem** is a novel memory framework designed to address the "destructive de-contextualization" problem in traditional RAG and long-context systems. Instead of compressing memory into static embeddings or graphs, E-mem shifts the paradigm to **Episodic Context Reconstruction**.

Inspired by biological engrams, E-mem employs a **Heterogeneous Hierarchical Architecture**:
1.  **Master Agent**: Orchestrates global planning and synthesizes reasoning.
2.  **Assistant Agents**: Maintain uncompressed memory chunks and perform local reasoning within activated segments to extract context-aware evidence.

By combining a **Multi-Pathway Routing** mechanism with distributed agentic reasoning, E-mem achieves State-of-the-Art performance on complex benchmarks (LoCoMo, HotpotQA) while significantly reducing token costs compared to full-context approaches.

## 🌟 Key Features

* **🧩 Episodic Context Reconstruction**: Unlike passive retrieval, Assistant agents actively reason within raw memory contexts to preserve sequential dependencies and logical integrity.
* **🤖 Master-Assistant Architecture**: Decouples high-level planning from low-level memory retention, enabling scalable "System 2" reasoning over extended horizons.
* **🛣️ Multi-Pathway Routing**: Implements a robust routing policy combining three orthogonal signals:
    * **Global Alignment**: Summary-based intent filtering.
    * **Semantic Association**: High-dimensional vector similarity.
    * **Symbolic Trigger**: Precise keyword/entity matching.
* **⚡ Latent State Optimization (KV Cache)**: Supports caching internal neural representations (KV tensors) for Assistant agents. This minimizes re-encoding overhead during memory activation, offering a trade-off between storage and latency.
* **🔌 Flexible Storage Modes**:
    * **Optimization Mode**: Uses local/cached tensors for maximum performance.
    * **Text Mode**: A lightweight fallback for API-based debugging or cloud inference.

## 📦 Installation

We recommend using `uv` for dependency management.

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

Get up and running with the E-mem factory. This example initializes a Master-Assistant setup.

```python
from src.conversation_manager import create_chat_manager

# 1. Initialize E-mem Manager
# The manager handles the coordination between the Master Agent and Assistant Agents.
manager = create_chat_manager(
    storage_mode="kv_cache",           # Enable Latent State Optimization (or use "text" for pure API)
    model_id="Qwen/Qwen3-4B",          # Assistant Agent Model (SLM)
    openai_config={                    # Master Agent Configuration
        "api_key": "your-key",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini"         # Master Agent (Global Planner)
    }
)

# 2. Add Memory (Memory Building)
# The system performs chunking and distributes contexts to Assistant Agents.
manager.chat("My name is Alice and I am a software engineer.", auto_save=True)

# 3. Query Memory (Reconstruction & Reasoning)
# Triggers Multi-Pathway Routing -> Assistant Local Reasoning -> Master Synthesis.
response = manager.chat("What is my profession?")
print(response)
# Output: "You are a software engineer."
```

## 📚 Documentation

Detailed documentation for researchers and engineers.

| Resource | Description |
|:---|:---|
| **[Architecture](docs/ARCHITECTURE.md)** | Deep dive into the Master-Assistant design and Routing mechanisms. |
| **[Operational Guides](docs/GUIDES.md)** | Guides on Persistence, Latent State Caching, and Model Compatibility. |
| **[API Reference](docs/API_REFERENCE.md)** | Complete API specification for Managers, Handlers, and Config. |

## 📁 Project Structure

```
mem-with-kv-cache/
├── src/
│   ├── agent/                    # Base agent with tool calling
│   ├── config/                   # Pydantic configuration schemas
│   ├── conversation_manager/     # Chat interface & factory
│   │   ├── base_chat_manager.py  # Shared base class
│   │   ├── chat_handler.py       # Implementation of memory reconstruction logic
│   │   └── factory.py            # Factory function
│   ├── memory/
│   │   ├── core/                 # Memory handlers
│   │   ├── kv_block_manager/     # Latent state (KV) storage implementation
│   │   ├── memory_agent/         # Assistant Agent implementation
│   │   └── router/               # Multi-Pathway Routing (Global/Semantic/Symbolic)
│   └── utils/                    # Utilities & prompts
├── evaluation/
│   ├── locomo/                   # LoCoMo benchmark scripts
│   └── hotpotqa/                 # HotpotQA benchmark scripts
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

To reproduce the results presented in our paper (ICML 2026 Submission), we provide automated scripts for both **LoCoMo** and **HotpotQA** benchmarks.

### LoCoMo Benchmark
Evaluates long-term memory coherence, including multi-hop and temporal reasoning tasks.

```bash
# Run LoCoMo evaluation
bash scripts/eval_locomo.sh
```

### HotpotQA Benchmark
Stress-tests the system on multi-hop QA with ultra-long contexts (streaming setting).

```bash
# Run HotpotQA evaluation
bash scripts/eval_hotpotqa.sh
```

> Detailed evaluation configurations and baseline comparisons can be found in [`evaluation/locomo/README.md`](evaluation/locomo/README.md) and [`evaluation/hotpotqa/README.md`](evaluation/hotpotqa/README.md).

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
