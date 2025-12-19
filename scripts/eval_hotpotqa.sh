#!/bin/bash
# HotpotQA Evaluation Runner Script

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get project root (parent of scripts/)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT" || exit 1

# Run evaluation from project root, passing all arguments
uv run python evaluation/hotpotqa/eval_hotpotqa.py \
    --config "./evaluation/hotpotqa/config.yaml" \
    "$@"

echo "Evaluation complete!"
