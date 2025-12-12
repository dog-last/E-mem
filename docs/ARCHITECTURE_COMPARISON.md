# Architecture Comparison: KV Cache vs Text Storage

## 系统架构对比

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Application Layer                        │
│                                                                   │
│  create_chat_manager(storage_mode="kv_cache" | "text")          │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌─────────────────┐            ┌──────────────────────┐
│  ChatManager    │            │ TextStorageChatManager│
│  (KV Cache)     │            │   (Text Storage)      │
└────────┬────────┘            └──────────┬───────────┘
         │                                │
         │ tools: add_memory, query_memory│
         │                                │
         ▼                                ▼
┌─────────────────┐            ┌──────────────────────┐
│ MemoryHandler   │            │ TextMemoryHandler    │
└────────┬────────┘            └──────────┬───────────┘
         │                                │
    ┌────┴────┐                      ┌────┴────┐
    │         │                      │         │
    ▼         ▼                      ▼         ▼
┌────────┐ ┌────────┐          ┌────────┐ ┌────────┐
│AddHandler│QueryHandler│      │TextAdd │ │TextQuery│
│        │ │        │          │Handler │ │Handler │
└───┬────┘ └───┬────┘          └───┬────┘ └───┬────┘
    │          │                   │          │
    ▼          ▼                   ▼          ▼
┌────────┐ ┌────────┐          ┌────────┐ ┌────────┐
│Memory  │ │ Router │          │Text    │ │ Router │
│Agent   │ │        │          │Memory  │ │(shared)│
│        │ │        │          │Agent   │ │        │
└───┬────┘ └────────┘          └───┬────┘ └────────┘
    │                              │
    ▼                              ▼
┌────────┐                    ┌────────┐
│KVBlock │                    │TextBlock│
│        │                    │        │
└───┬────┘                    └───┬────┘
    │                              │
    ▼                              ▼
┌────────┐                    ┌────────┐
│kv_data/│                    │text_data/│
│ *.pt   │                    │ *.json │
└────────┘                    └────────┘
```

## 组件对应关系

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

## 数据流对比

### KV Cache Mode - Add Memory
```
User Input
    ↓
ChatManager.chat(auto_save=True)
    ↓
MemoryHandler.add_memory(text)
    ↓
AddHandler.add_memory(text)
    ↓
MemoryAgent.add([text])
    ↓
MemoryAgent._add_knowledge([text])
    ↓
[Tokenize → Forward Pass → Generate KV Cache]
    ↓
KVBlock.save_cache(cache_state)
    ↓
Save to kv_data/*.pt
```

### Text Storage Mode - Add Memory
```
User Input
    ↓
TextStorageChatManager.chat(auto_save=True)
    ↓
TextMemoryHandler.add_memory(text)
    ↓
TextAddHandler.add_memory(text)
    ↓
TextMemoryAgent.add([text])
    ↓
[Tokenize for counting only]
    ↓
TextBlock.add_chunk(text, token_count)
    ↓
Save to text_data/*.json
```

### KV Cache Mode - Query Memory
```
User Query
    ↓
ChatManager.chat(query)
    ↓
MemoryHandler.query_memory(query)
    ↓
[Parallel: QueryHandler + AddHandler]
    ↓
Router.map_reduce_blocks(query)
    ↓
[Select relevant agents via LLM]
    ↓
MemoryAgent.query(question)
    ↓
[Load KV Cache → Generate with cache]
    ↓
Return results
```

### Text Storage Mode - Query Memory
```
User Query
    ↓
TextStorageChatManager.chat(query)
    ↓
TextMemoryHandler.query_memory(query)
    ↓
[Parallel: TextQueryHandler + TextAddHandler]
    ↓
Router.map_reduce_blocks(query)
    ↓
[Select relevant agents via LLM]
    ↓
TextMemoryAgent.query(question)
    ↓
[Load text → LLM API call with full context]
    ↓
Return results
```

## 关键差异

### 1. 存储机制
- **KV Cache**: 存储PyTorch张量（keys, values）
- **Text Storage**: 存储纯文本JSON

### 2. 查询机制
- **KV Cache**: 加载缓存 → GPU推理
- **Text Storage**: 加载文本 → API调用

### 3. 资源需求
- **KV Cache**: GPU显存 + 计算
- **Text Storage**: API配额 + 网络

### 4. 速度特性
- **KV Cache**: 快速预填充（缓存复用）
- **Text Storage**: 每次完整编码

### 5. 可维护性
- **KV Cache**: 二进制文件，难以调试
- **Text Storage**: JSON文件，易于检查

## 共享组件

### Router
- 两种模式共享同一个Router实现
- 基于LLM的智能路由
- 使用summary进行块选择

### 工具接口
- 两种模式暴露相同的工具：
  - `add_memory(memory: str)`
  - `query_memory(query: str)`

### Overlap机制
- 两种模式都支持块间重叠
- 默认10%重叠率
- 保证上下文连续性

## 选择指南

```
需要GPU? ──No──> Text Storage
    │
   Yes
    │
    ▼
速度优先? ──Yes──> KV Cache
    │
   No
    │
    ▼
易调试? ──Yes──> Text Storage
    │
   No
    │
    ▼
    KV Cache
```

## 性能对比（估算）

| Metric | KV Cache | Text Storage |
|--------|----------|--------------|
| Add Memory | ~100ms | ~50ms |
| Query (cold) | ~500ms | ~1000ms |
| Query (warm) | ~200ms | ~1000ms |
| GPU Memory | ~4GB | 0GB |
| Disk Space | ~100MB/block | ~1MB/block |
| Debuggability | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Deployment | ⭐⭐ | ⭐⭐⭐⭐⭐ |

## 代码复用率

- **共享代码**: Router, BaseAgent, Prompts
- **独立代码**: Block, Agent, Handler, ChatManager
- **复用率**: ~30% (Router + 基础设施)

## 总结

两种模式在架构上完全对等，提供相同的功能和接口，用户可以根据资源和需求灵活选择。
