# HotpotQA Evaluation

Evaluation script for HotpotQA dataset using the KV-Cached Memory Agent System.

## Setup

1. Copy the example config:
```bash
cd evaluation/hotpotqa
cp config.example.yaml config.yaml
```

2. Edit `config.yaml` with your settings:
   - Set your OpenAI API key and base URL
   - Configure model settings
   - Set dataset path

## Dataset Format

HotpotQA dataset should be in JSON format:
```json
[
  {
    "index": 0,
    "context": "Long context text...",
    "input": "Question text?",
    "answers": ["answer1", "answer2"]
  }
]
```

## Usage

### Using Shell Script (WSL/Linux)
```bash
# Run with default config
bash scripts/run_hotpotqa_eval.sh

# Run with custom config
bash scripts/run_hotpotqa_eval.sh --config evaluation/hotpotqa/config.yaml

# Run with ratio
bash scripts/run_hotpotqa_eval.sh --ratio 0.1
```

### Direct Python
```bash
python evaluation/hotpotqa/eval_hotpotqa.py --config evaluation/hotpotqa/config.yaml
```

## How It Works

For each sample:
1. **Create Agent**: Initialize ChatManager with `clean_cache_first=True`
2. **Add Context**: Use `add_memory()` to store the context
3. **Query**: Use `search_memory()` to answer the question
4. **Cleanup**: Delete agent and clear GPU cache to avoid OOM

## Metrics

- **F1 Score**: Token-level F1 between prediction and ground truth
- **Exact Match (EM)**: Binary score for exact match after normalization

## Output

Results are saved to `evaluation/hotpotqa/results/hotpotqa_eval_<timestamp>.json`:
```json
{
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "total_samples": 100,
  "metrics": {
    "f1": 0.75,
    "em": 0.60
  },
  "individual_results": [...]
}
```

Logs are saved to `evaluation/hotpotqa/logs/hotpotqa_eval_<timestamp>.log`
