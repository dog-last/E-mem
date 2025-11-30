# Architecture Documentation

## System Overview

This system implements a novel memory management approach for conversational AI using KV cache instead of traditional RAG retrieval.

## Core Innovation

### Problem with Traditional RAG
- Embedding-based retrieval can miss semantic nuances
- Chunking loses context
- Requires external vector databases
- Retrieval accuracy depends on embedding quality

### Our Solution
- **Full Context**: LLM sees complete memory blocks, not fragments
- **KV Cache Reuse**: Faster prefilling by reusing cached key-value pairs
- **LLM-based Routing**: Use LLM to select relevant memory blocks via summaries
- **Parallel Agents**: Multiple agents handle different memory blocks concurrently

## Component Details

### 1. KVBlock (`src/memory/kv_block_manager/block.py`)

**Purpose**: Persistent storage for KV cache

**Key Features**:
- Stores cache as PyTorch tensors on disk
- Tracks token usage and capacity
- Auto-creates `kv_data/` in current working directory
- Thread-safe operations

**Storage Format**:
```python
{
    "global_offset": int,      # Total tokens processed
    "saved_chunks": [          # List of cache chunks
        {
            "cache": [(k, v), ...],  # Per-layer KV tensors
            "start": int,            # Starting position
            "length": int            # Chunk length
        }
    ],
    "chunk_number": int,       # Number of chunks
    "model_id": str           # Model identifier
}
```

### 2. MemoryAgent (`src/memory/memory_agent/agent.py`)

**Purpose**: Manages a single memory block with KV cache

**Lifecycle**:
1. **Active State**: Accepts new memories
   - Incrementally builds KV cache
   - Each new chunk sees all previous cache (critical for coherence)
   - Stores only NEW cache per chunk (memory efficient)
   
2. **Inactive State**: Query-only
   - Triggered when block reaches 90% capacity
   - Generates summary for router
   - No longer accepts new memories

**Key Methods**:
- `add(text_chunks)`: Add memories incrementally
- `query(question)`: Generate response using cached context
- `_create_summaries()`: Generate block summary when full

**Technical Details**:
- Uses `DynamicCache` from transformers
- Correct position IDs for RoPE embeddings
- Repetition penalty during generation
- Removes `<thinking>` tags from output

### 3. Router (`src/memory/router/router.py`)

**Purpose**: LLM-based selection of relevant memory blocks

**How It Works**:
1. Receives user query
2. Formats query with all memory summaries
3. LLM returns indices of relevant blocks
4. Parallel query execution using ThreadPoolExecutor

**Input Format**:
```xml
<query>User question here</query>
<summary_list>
    <summary>
        <index>0</index>
        <content>Summary of block 0</content>
    </summary>
    <summary>
        <index>1</index>
        <content>Summary of block 1</content>
    </summary>
</summary_list>
```

**Output Format**:
```xml
<summary_index>0,2,5</summary_index>
```

### 4. MemoryHandler (`src/memory/core/loop_handler.py`)

**Purpose**: Orchestrates memory operations

**Components**:
- **AddHandler**: Manages active memory agent
  - Creates new agent when current becomes full
  - Handles memory addition
  
- **QueryHandler**: Manages inactive agents via router
  - Parallel query to selected blocks
  - Aggregates results

**Memory Addition Flow**:
```
User adds memory
    ↓
AddHandler.add_memory()
    ↓
MemoryAgent.add()
    ↓
Block full? → Yes → Create summary
              ↓       Move to inactive
              ↓       Create new active agent
              ↓       Register with router
              No → Continue
```

**Query Flow**:
```
User query
    ↓
Parallel execution:
    ├─→ QueryHandler (inactive agents)
    │       ↓
    │   Router selects relevant blocks
    │       ↓
    │   Parallel query selected agents
    │
    └─→ AddHandler (active agent)
            ↓
        Query recent memories
            ↓
Merge results:
"Old memory: ... \n New memory: ..."
```

### 5. ChatManager (`src/conversation_manager/chat_handler.py`)

**Purpose**: User-facing interface with tool calling

**Tools**:
1. **add_memory**: Store information
   - Agent decides what to extract and store
   - Optional: save original input vs processed

2. **query_memory**: Retrieve information
   - Searches both active and inactive agents
   - Returns aggregated results

**Tool Calling Flow**:
```
User input
    ↓
LLM decides: Need tools?
    ↓
Yes → Execute tools (max_tool_rounds)
    ↓
    Tool results added to context
    ↓
    LLM generates final response
    ↓
No → Direct response
    ↓
Reset conversation (no history kept)
```

## Data Flow Example

### Adding Memory

```
User: "I love hiking in the mountains"
    ↓
ChatManager.chat()
    ↓
LLM: [tool_call: add_memory("User enjoys hiking in mountains")]
    ↓
MemoryHandler.add_memory()
    ↓
AddHandler.add_memory()
    ↓
MemoryAgent.add(["User enjoys hiking in mountains"])
    ↓
Format: "<|im_start|>system\n...<|im_start|>user\n[Context 1]\nUser enjoys hiking..."
    ↓
Tokenize → Forward pass → Extract KV cache
    ↓
KVBlock.save_cache()
    ↓
Save to: kv_data/kv_cache_{uuid}_{timestamp}.pt
```

### Querying Memory

```
User: "What are my hobbies?"
    ↓
ChatManager.chat()
    ↓
LLM: [tool_call: query_memory("user hobbies")]
    ↓
MemoryHandler.query_memory()
    ↓
Parallel:
    ├─→ QueryHandler.query_memory()
    │       ↓
    │   Router.map_reduce_blocks()
    │       ↓
    │   LLM selects: [0, 2] (relevant blocks)
    │       ↓
    │   ThreadPool: [Agent0.query(), Agent2.query()]
    │       ↓
    │   Merge KV cache → Generate responses
    │
    └─→ AddHandler.query_new_agent()
            ↓
        Active agent queries recent memory
            ↓
Combine: "Old: ... New: ..."
    ↓
Return to LLM
    ↓
LLM: "Based on the memory, you enjoy hiking in mountains."
```

## Performance Considerations

### Memory Usage
- **KV Cache Size**: ~2 bytes per token per layer (bfloat16)
- **Example**: 32K tokens, 32 layers = ~2GB per block
- **Mitigation**: Store on disk, load on demand

### Speed
- **Prefilling**: KV cache reuse → 2-5x faster than re-encoding
- **Parallel Queries**: ThreadPoolExecutor → N agents in parallel
- **Trade-off**: Initial cache building is slower

### Scalability
- **Blocks**: Each block = 90% of context window
- **Example**: 32K window → 28.8K per block
- **10 blocks** = ~288K tokens of memory
- **Router overhead**: Linear with number of blocks

## Limitations

1. **Not for Massive Scale**: Vector DBs better for millions of documents
2. **Cache Rebuild Cost**: Adding memory requires forward pass
3. **Context Window Bound**: Limited by model's max context
4. **LLM Routing Cost**: Router call per query

## Future Improvements

1. **Hierarchical Routing**: Multi-level summaries for more blocks
2. **Incremental Summaries**: Update summaries without full regeneration
3. **Compression**: Quantize KV cache for smaller storage
4. **Async Operations**: Non-blocking memory addition
5. **Multi-modal**: Extend to images/audio with vision models

## Comparison: This System vs Traditional RAG

| Aspect | This System | Traditional RAG |
|--------|-------------|-----------------|
| Retrieval | LLM-based routing | Embedding similarity |
| Context | Full memory blocks | Retrieved chunks |
| Accuracy | High (full context) | Depends on retrieval |
| Speed | Fast (KV reuse) | Fast (vector search) |
| Scale | Limited (context window) | Unlimited (vector DB) |
| Components | LLM only | LLM + Embeddings + VectorDB |
| Use Case | Personal memory | Knowledge bases |

## Conclusion

This system is ideal for **conversational AI with limited, user-specific memory** where accuracy and context understanding are critical. It trades scalability for precision by using LLMs throughout the pipeline and leveraging KV cache for efficiency.
