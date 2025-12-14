# HotpotQA Evaluation Implementation

## Overview
This implementation adapts the HotpotQA evaluation to use the KV-Cached Memory Agent System's native APIs (`add_memory` and `query_memory`).

## Key Design Decisions

### 1. Direct API Usage
- Uses `agent.add_memory()` to store context
- Uses `agent.search_memory()` to query and answer questions
- No intermediate GAM framework components

### 2. Clean Cache Management
- Each sample starts with `clean_cache_first=True`
- Agent is deleted after each sample
- GPU cache is explicitly cleared with `torch.cuda.empty_cache()`
- Prevents OOM errors during long evaluation runs

### 3. Configuration
- Uses local `config.yaml` in `evaluation/hotpotqa/` directory
- Follows same pattern as `evaluation/locomo/`
- Supports both KV cache and text storage modes

### 4. Metrics
- F1 Score: Token-level overlap between prediction and ground truth
- Exact Match (EM): Binary score after normalization
- Normalization removes articles, punctuation, and extra whitespace

## File Structure

```
evaluation/hotpotqa/
├── config.example.yaml       # Configuration template
├── eval_hotpotqa.py          # Main evaluation script
├── README.md                 # Usage documentation
└── IMPLEMENTATION.md         # This file

evaluation/eval_data/hotpotqa/
├── eval_50.json              # 50 samples
├── eval_400.json             # 400 samples
└── eval_1600.json            # 1600 samples

scripts/
└── run_hotpotqa_eval.sh      # Bash runner script
```

## Workflow Per Sample

1. **Initialize**: Create ChatManager with `clean_cache_first=True`
2. **Add Context**: Call `agent.add_memory(context)`
3. **Query**: Call `agent.search_memory(query_prompt)` with question
4. **Evaluate**: Calculate F1 and EM scores
5. **Cleanup**: Delete agent and clear GPU cache

## Differences from eval_locomo.py

| Aspect | eval_locomo.py | eval_hotpotqa.py |
|--------|----------------|------------------|
| Dataset | Conversational QA | Multi-hop QA |
| Context | Multiple conversation turns | Single long context |
| Memory Addition | Multiple `chat()` calls | Single `add_memory()` call |
| Query | Single question per sample | Single question per sample |
| Metrics | Category-specific prompts | F1 + EM scores |

## Running the Evaluation

```bash
# Copy config
cd evaluation/hotpotqa
cp config.example.yaml config.yaml
# Edit config.yaml with your settings

# Run evaluation
bash ../../scripts/run_hotpotqa_eval.sh

# Or run directly
python evaluation/hotpotqa/eval_hotpotqa.py --config evaluation/hotpotqa/config.yaml
```

## Memory Management

The implementation ensures proper resource cleanup:
- Agent deletion after each sample
- Explicit GPU cache clearing
- No persistent state between samples
- Prevents memory leaks and OOM errors

## Future Enhancements

Potential improvements:
- Batch processing for faster evaluation
- Parallel sample processing
- Caching of common contexts
- Support for multi-hop reasoning analysis
