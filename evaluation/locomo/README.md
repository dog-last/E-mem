# LoComo Evaluation

Evaluation framework for testing the memory system on the LoComo dataset.

## Structure

```
evaluation/locomo/
├── eval_locomo.py       # Main evaluation script
├── load_dataset.py      # Dataset loader
├── utils.py             # Evaluation metrics
├── eval_data/           # Dataset files
├── logs/                # Logs (auto-created, gitignored)
├── results/             # Results (auto-created, gitignored)
├── kv_data/             # KV cache (auto-created, gitignored)
└── text_data/           # Text storage (auto-created, gitignored)
```

## Configuration

Edit `config.yaml` in the project root to configure:

- **Model settings**: model_id, context window, device settings
- **Memory settings**: storage mode, cache behavior, router prompts
- **Evaluation settings**: dataset path, output directory, ratio
- **Auto-save settings**: 
  - `conversation_auto_save`: Enable/disable auto-save for conversation turns
  - QA questions NEVER use auto_save (hardcoded)

## Usage

### Using the bash script (recommended):

```bash
# Run from project root
bash scripts/run_eval.sh

# With custom config
bash scripts/run_eval.sh --config config.yaml

# Override specific settings
bash scripts/run_eval.sh --model_id "Qwen/Qwen3-1.7B" --ratio 0.1

# Enable auto-save for conversation turns
bash scripts/run_eval.sh --conversation_auto_save

# Full example
bash scripts/run_eval.sh \
    --config config.yaml \
    --model_id "Qwen/Qwen3-1.7B" \
    --dataset "evaluation/locomo/eval_data/locomo10.json" \
    --ratio 0.5 \
    --conversation_auto_save
```

### Using Python directly:

```bash
# Run from project root
python evaluation/locomo/eval_locomo.py

# With arguments
python evaluation/locomo/eval_locomo.py \
    --config config.yaml \
    --model_id "Qwen/Qwen3-1.7B" \
    --ratio 0.1 \
    --conversation_auto_save
```

## Arguments

- `--config`: Path to config file (default: `config.yaml`)
- `--model_id`: Override model ID from config
- `--dataset`: Override dataset path from config
- `--ratio`: Override evaluation ratio (0.0-1.0)
- `--conversation_auto_save`: Enable auto-save for conversation turns (flag)

## Auto-Save Behavior

**Important**: The system has different auto-save behavior for conversations vs QA:

1. **Conversation turns**: Can use auto-save if `conversation_auto_save` is enabled
   - When enabled: Directly saves conversation text without LLM processing
   - When disabled: LLM decides whether to save and how to process

2. **QA questions**: NEVER use auto-save (hardcoded in eval script)
   - Always uses normal chat mode where LLM processes the question
   - This ensures proper question answering behavior

## Output

Results are saved to:
- `evaluation/locomo/results/locomo_eval_<timestamp>.json` - Evaluation results
- `evaluation/locomo/logs/eval_<timestamp>.log` - Detailed logs

## Metrics

The evaluation calculates:
- Exact Match
- F1 Score
- ROUGE (1, 2, L)
- BLEU (1, 2, 3, 4)
- METEOR

Metrics are aggregated:
- Overall (all questions)
- Per category (1-5)

## Categories

LoComo dataset has 5 question categories:
1. Factual questions
2. Temporal questions
3. Reasoning questions
4. Detailed questions
5. Adversarial questions

Configure which categories to evaluate in `config.yaml`.
