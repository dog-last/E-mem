# LoComo Evaluation

Evaluation framework for testing the memory system on the LoComo dataset.

## Structure

```
evaluation/
├── config.yaml          # Configuration file
├── eval_locomo.py       # Main evaluation script
├── load_dataset.py      # Dataset loader
├── utils.py             # Evaluation metrics
├── run_eval.sh          # Bash script for running evaluation
├── test_setup.py        # Setup verification script
├── logs/                # Evaluation logs (auto-created)
└── results/             # Evaluation results (auto-created)
```

**Note**: The `logs/` and `results/` directories are automatically created by the evaluation script when needed.

## Configuration

Edit `config.yaml` to configure:

- **Model settings**: model_id, context window, device settings
- **Memory settings**: cache behavior, router prompts
- **Evaluation settings**: dataset path, output directory, ratio
- **Auto-save settings**: 
  - `conversation_auto_save`: Enable/disable auto-save for conversation turns
  - QA questions NEVER use auto_save (hardcoded)

## Setup

First, verify your setup:

```bash
uv run python evaluation/test_setup.py
```

This will check if all dependencies are installed and directories can be created.

## Usage

### Using the bash script (recommended for WSL):

```bash
# Basic usage with default config
bash evaluation/run_eval.sh

# With custom config
bash evaluation/run_eval.sh --config evaluation/config.yaml

# Override specific settings
bash evaluation/run_eval.sh --model_id "Qwen/Qwen2.5-7B-Instruct" --ratio 0.1

# Enable auto-save for conversation turns
bash evaluation/run_eval.sh --conversation_auto_save

# Full example
bash evaluation/run_eval.sh \
    --config evaluation/config.yaml \
    --model_id "Qwen/Qwen2.5-7B-Instruct" \
    --dataset "../locomo10_origin.json" \
    --ratio 0.5 \
    --conversation_auto_save
```

### Using Python directly:

```bash
# Basic usage
uv run python evaluation/eval_locomo.py

# With arguments
uv run python evaluation/eval_locomo.py \
    --config evaluation/config.yaml \
    --model_id "Qwen/Qwen2.5-7B-Instruct" \
    --ratio 0.1 \
    --conversation_auto_save
```

## Arguments

- `--config`: Path to config file (default: `evaluation/config.yaml`)
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
- `evaluation/results/locomo_eval_<timestamp>.json` - Evaluation results
- `evaluation/logs/eval_<timestamp>.log` - Detailed logs

## Metrics

The evaluation calculates:
- Exact Match
- F1 Score
- ROUGE (1, 2, L)
- BLEU (1, 2, 3, 4)
- METEOR
- Sentence-BERT Similarity

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
