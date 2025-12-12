# Architecture Comparison: KV Cache vs Text Storage

## Component Mapping

| Layer | KV Cache Mode | Text Storage Mode | Shared |
|-------|---------------|-------------------|--------|
| **Interface** | ChatManager | TextStorageChatManager | ✗ |
| **Orchestrator** | MemoryHandler | TextMemoryHandler | ✗ |
| **Add Logic** | AddHandler | TextAddHandler | ✗ |
| **Query Logic** | QueryHandler | TextQueryHandler | ✗ |
| **Agent** | MemoryAgent | TextMemoryAgent | ✗ |
| **Router** | Router | Router | ✓ |
| **Block** | KVBlock | TextBlock | ✗ |
| **Storage** | kv_data/*.pt | text_data/*.json | ✗ |

## Key Differences

### Storage Mechanism
- **KV Cache**: Stores PyTorch tensors (keys, values) in `.pt` files
- **Text Storage**: Stores plain text in JSON files

### Query Mechanism
- **KV Cache**: Loads cached KV tensors → GPU inference with model
- **Text Storage**: Loads text → LLM API call with full context

### Resource Requirements
- **KV Cache**: Requires GPU with VRAM
- **Text Storage**: Requires API access (OpenAI, etc.)

### Speed Characteristics
- **KV Cache**: Uses cached KV tensors
- **Text Storage**: Full text re-encoding on each query

### Maintainability
- **KV Cache**: Binary files, harder to debug
- **Text Storage**: JSON files, human-readable and easy to inspect

## Shared Components

### Router
- Both modes use the same Router implementation
- LLM-based intelligent routing
- Selects relevant memory blocks using summaries

### Tool Interface
Both modes expose identical tools:
- `add_memory(memory: str)`
- `query_memory(query: str)`

### Overlap Mechanism
- Both modes support inter-block overlap
- Default 10% overlap ratio
- Ensures context continuity between blocks

## When to Use Each Mode

### Use KV Cache Mode When:
- GPU resources are available
- Local deployment with GPU
- Working with conversational AI
- Local deployment with GPU

### Use Text Storage Mode When:
- No GPU available or limited GPU memory
- Using cloud LLM APIs (OpenAI, Anthropic, etc.)
- Easier debugging needed (human-readable storage)
- Simpler deployment required
- Cost-effective for low-frequency queries

## Summary

Both modes are architecturally equivalent, providing the same functionality and interface. Users can choose based on available resources and deployment requirements.
