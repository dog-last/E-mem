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

[cite_start]**E-mem** is a novel memory framework designed to address the "destructive de-contextualization" problem in traditional RAG and long-context systems[cite: 60]. [cite_start]Instead of compressing memory into static embeddings or graphs, E-mem shifts the paradigm to **Episodic Context Reconstruction**[cite: 62].

[cite_start]Inspired by biological engrams, E-mem employs a **Heterogeneous Hierarchical Architecture**[cite: 63]:
1.  [cite_start]**Master Agent**: Orchestrates global planning and synthesizes reasoning[cite: 63].
2.  [cite_start]**Assistant Agents**: Maintain uncompressed memory chunks and perform local reasoning within activated segments to extract context-aware evidence[cite: 64].

[cite_start]By combining a **Multi-Pathway Routing** mechanism with distributed agentic reasoning, E-mem achieves State-of-the-Art performance on complex benchmarks (LoCoMo, HotpotQA) while significantly reducing token costs compared to full-context approaches[cite: 65, 168].

## 🌟 Key Features

* [cite_start]**🧩 Episodic Context Reconstruction**: Unlike passive retrieval, Assistant agents actively reason within raw memory contexts to preserve sequential dependencies and logical integrity[cite: 166].
* [cite_start]**🤖 Master-Assistant Architecture**: Decouples high-level planning from low-level memory retention, enabling scalable "System 2" reasoning over extended horizons[cite: 162, 174].
* [cite_start]**🛣️ Multi-Pathway Routing**: Implements a robust routing policy combining three orthogonal signals[cite: 353]:
    * [cite_start]**Global Alignment**: Summary-based intent filtering[cite: 367].
    * [cite_start]**Semantic Association**: High-dimensional vector similarity[cite: 370].
    * [cite_start]**Symbolic Trigger**: Precise keyword/entity matching[cite: 373].
* **⚡ Latent State Optimization (KV Cache)**: Supports caching internal neural representations (KV tensors) for Assistant agents. [cite_start]This minimizes re-encoding overhead during memory activation, offering a trade-off between storage and latency[cite: 496].
* **🔌 Flexible Storage Modes**:
    * **Optimization Mode**: Uses local/cached tensors for maximum performance.
    * **Text Mode**: A lightweight fallback for API-based debugging or cloud inference.

## 📦 Installation

We recommend using `uv` for dependency management.

```bash
# Clone the repository
git clone [https://github.com/your-username/mem-with-kv-cache.git](https://github.com/your-username/mem-with-kv-cache.git)
cd mem-with-kv-cache

# Install with uv (Recommended)
uv sync

# Setup configuration
cp config.example.yaml config.yaml
# Edit config.yaml with your settings

## ⚡ Quick Start

Get up and running with the E-mem factory. This example initializes a Master-Assistant setup.Pythonfrom src.conversation_manager import create_chat_manager

# 1. Initialize E-mem Manager
# The manager handles the coordination between the Master Agent and Assistant Agents.
manager = create_chat_manager(
    storage_mode="kv_cache",           # Enable Latent State Optimization (or use "text" for pure API)
    model_id="Qwen/Qwen3-4B",          # Assistant Agent Model (SLM) [cite: 583]
    openai_config={                    # Master Agent Configuration
        "api_key": "your-key",
        "base_url": "[https://api.openai.com/v1](https://api.openai.com/v1)",
        "model": "gpt-4o-mini"         # Master Agent (Global Planner) [cite: 583]
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

##📚 Documentation

Detailed documentation for researchers and engineers.ResourceDescriptionArchitectureDeep dive into the Master-Assistant design and Routing mechanisms.Operational GuidesGuides on Persistence, Latent State Caching, and Model Compatibility.API ReferenceComplete API specification for Managers, Handlers, and Config.📁 Project Structuremem-with-kv-cache/
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

##🧪 Testing

Ensure system stability with our comprehensive test suite.Bash# Run all tests

pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_chat_manager.py -v

##📊 Evaluation (Reproducing Results)

To reproduce the results presented in our paper (ICML 2026 Submission), we provide automated scripts for both LoCoMo and HotpotQA benchmarks.LoCoMo BenchmarkEvaluates long-term memory coherence, including multi-hop and temporal reasoning tasks.Bash# Run LoCoMo evaluation
bash scripts/eval_locomo.sh
HotpotQA BenchmarkStress-tests the system on multi-hop QA with ultra-long contexts (streaming setting).Bash# Run HotpotQA evaluation
bash scripts/eval_hotpotqa.sh
Detailed evaluation configurations and baseline comparisons can be found in evaluation/locomo/README.md and evaluation/hotpotqa/README.md.🤝 ContributingWe welcome contributions to E-mem! Please follow this workflow to ensure quality:Fork & Clone: Fork the repository and clone it locally.Install Dependencies:Bashuv sync
Setup Pre-commit:Ensure code quality hooks are active.Bashpre-commit install
Run Tests:Verify that your changes don't break existing functionality.Bashpytest tests/
Update Documentation:If you modify the API or features, please update the relevant docs/*.md files or README.md file.📄 License & AttributionThis project is licensed under the Apache License 2.0. See LICENSE for details.AcknowledgementsThis project includes some evaluation code adapted from the GAM (General Agentic Memory) framework. We explicitly acknowledge and thank the original authors. Please see NOTICE for detailed attribution and license information regarding third-party code.
