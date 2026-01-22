# LoComo Evaluation

Evaluation E-mem using the LoCoMo dataset.

## Setup

1. **Dataset**: Download [`locomo10.json`](https://github.com/snap-research/locomo/blob/main/data/locomo10.json).
2. **Config**: Edit `config.yaml` in the `evaluation/locomo/` directory to configure:

- **Model settings**: model_id, context window, device settings
- **Memory settings**: storage mode, cache behavior, router prompts
- **Evaluation settings**: dataset path, output directory, ratio

## Usage

```bash
bash scripts/run_eval.sh

# Override specific settings
bash scripts/run_eval.sh --model_id "Qwen/Qwen3-1.7B" --ratio 0.1
```

## Implementation

For each conversation:
1. **Create Agent**: Initialize ChatManager with `clean_cache_first=True`
2. **Add Context**: Use `add_memory()` to store the conversation
3. **Query**: Use `search_memory()` to answer the question
4. **Cleanup**: Delete agent and clear GPU cache to avoid OOM
5. **Calculate Metrics**:
  - Exact Match
  - F1 score
  - ROUGE score (1/2/L)
  - BLEU score (1-4)
  - METEOR score

## Metrics & Output

- **Metrics**: Exact Match, F1, ROUGE (1/2/L), BLEU (1-4), and METEOR.
- **Results**: Aggregated overall and per category, saved to `evaluation/locomo/results/`.
- **Logs**: Saved to `evaluation/locomo/logs/`.

