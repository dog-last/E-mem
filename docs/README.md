# Documentation

## Quick Start

### KV Cache Mode (GPU Required)

```python
from src.conversation_manager.factory import create_chat_manager

manager = create_chat_manager(
    storage_mode="kv_cache",
    model_id="Qwen/Qwen3-4B",
    openai_config={"api_key": "your-key"}
)

manager.chat("My name is Alice.", auto_save=True)
response = manager.chat("What is my name?")
```

### Text Storage Mode (No GPU Required)

```python
manager = create_chat_manager(
    storage_mode="text",
    model_id="Qwen/Qwen3-4B",
    openai_config={"api_key": "your-key"}
)
```

See [QUICKSTART_TEXT_STORAGE.md](QUICKSTART_TEXT_STORAGE.md) for details.

## Project Structure

```
mem-with-kv-cache/
├── src/                    # Core source code
├── examples/               # Example scripts
├── evaluation/locomo/      # LoComo evaluation
├── scripts/                # Utility scripts (run_eval.sh)
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── config.yaml             # Main config (gitignored)
└── config.example.yaml     # Config template
```

## Configuration

Copy and edit the config file:

```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your settings
```

### Key Settings

```yaml
model:
  model_id: "Qwen/Qwen3-4B"
  openai_config:
    api_key: "your-api-key"
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o-mini"

memory:
  storage_mode: "kv_cache"  # or "text"
  clean_cache_first: true
  overlap_ratio: 0.1
  max_concurrent_gpu_operations: 2

evaluation:
  dataset_path: "evaluation/locomo/eval_data/locomo10.json"
  output_dir: "evaluation/locomo/results"
  ratio: 1.0
  categories: [1, 2, 3, 4, 5]
```

## Running Examples

```bash
# Quick start
python examples/quickstart.py

# Text storage mode
python examples/example_text_storage.py

# Simple example
python examples/example_simple.py
```

## Running Evaluation

```bash
# Basic
bash scripts/run_eval.sh

# With options
bash scripts/run_eval.sh --ratio 0.1 --conversation_auto_save
```

## Running Tests

```bash
pytest tests/
```

## Import Paths

```python
# Core
from src.conversation_manager.factory import create_chat_manager
from src.conversation_manager.chat_handler import ChatManager

# Evaluation
from evaluation.locomo.load_dataset import load_locomo_dataset
from evaluation.locomo.utils import calculate_metrics

# Config
from config import MAX_CONCURRENT_GPU_OPERATIONS
```

## Additional Documentation

- [QUICKSTART_TEXT_STORAGE.md](QUICKSTART_TEXT_STORAGE.md) - Text storage quick start
- [TEXT_STORAGE_README.md](TEXT_STORAGE_README.md) - Text storage details
- [ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md) - KV Cache vs Text Storage
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Implementation details
- [PERSISTENCE.md](PERSISTENCE.md) - KV cache persistence
- [../evaluation/locomo/README.md](../evaluation/locomo/README.md) - Evaluation guide

## Common Tasks

### Create Chat Manager from Config

```python
import yaml
from src.conversation_manager.factory import create_chat_manager

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

manager = create_chat_manager(
    storage_mode=config['memory']['storage_mode'],
    model_id=config['model']['model_id'],
    openai_config=config['model']['openai_config'],
    clean_cache_first=config['memory']['clean_cache_first']
)
```

### Run Evaluation Programmatically

```python
from evaluation.locomo.eval_locomo import evaluate_dataset, setup_logger
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logger = setup_logger('eval.log')
results = evaluate_dataset(config, logger)
```

## Troubleshooting

### Config not found
```bash
cp config.example.yaml config.yaml
```

### Import errors
Ensure you're in project root when running scripts.

### GPU memory issues
Use text storage mode or reduce `model_context_window` in config.
