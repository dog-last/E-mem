#!/bin/bash
# Evaluation script for LoComo dataset

# Default values
CONFIG="evaluation/config.yaml"
MODEL_ID=""
DATASET=""
RATIO=""
CONVERSATION_AUTO_SAVE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --model_id)
            MODEL_ID="--model_id $2"
            shift 2
            ;;
        --dataset)
            DATASET="--dataset $2"
            shift 2
            ;;
        --ratio)
            RATIO="--ratio $2"
            shift 2
            ;;
        --conversation_auto_save)
            CONVERSATION_AUTO_SAVE="--conversation_auto_save"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run evaluation
uv run python eval_locomo.py \
    --config "$CONFIG" \
    $MODEL_ID \
    $DATASET \
    $RATIO \
    $CONVERSATION_AUTO_SAVE

echo "Evaluation complete!"
