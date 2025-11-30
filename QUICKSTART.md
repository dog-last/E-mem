# Quick Start Guide

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- OpenAI API key (for router agent)
- Local LLM or HuggingFace model access

## Installation

### 1. Clone and Install Dependencies

```bash
git clone <your-repo-url>
cd mem-with-kv-cache
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config.example.py config.py
```

Edit `config.py`:

```python
# Set your model path
MODEL_ID = "Qwen/Qwen3-0.6B"  # or local path

# Set OpenAI API key
OPENAI_CONFIG = {
    "api_key": "sk-your-key-here",
}

# Optional: Enable quantization for lower memory usage
USE_QUANTIZATION = True
QUANTIZATION_BITS = 4
```

## Usage

### Option 1: Interactive Chat (Recommended)

```bash
python main.py
```

Example conversation:
```
You: My name is Alice and I work as a data scientist.
Assistant: [Stores memory and responds]

You: What's my name?
Assistant: Your name is Alice.

You: What do I do for work?
Assistant: You work as a data scientist.
```

### Option 2: Simple Memory Test

```bash
python example_simple.py
```

This demonstrates basic memory addition and querying without the chat interface.

### Option 3: Programmatic Usage

```python
from src.conversation_manager.chat_handler import ChatManager

# Initialize
chat = ChatManager(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    openai_config={"api_key": "your-key"}
)

# Add memory
response = chat.chat(
    user_input="I love Python programming.",
    auto_save=False  # Let agent decide
)

# Query memory
response = chat.chat(
    user_input="What programming language do I like?"
)
print(response)
```

## Common Issues

### Issue 1: Out of Memory

**Solution**: Enable quantization

```python
# In config.py
USE_QUANTIZATION = True
QUANTIZATION_BITS = 4
```

### Issue 2: Model Not Found

**Solution**: Download model first

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

### Issue 3: OpenAI API Error

**Solution**: Check API key and endpoint

```python
# Test OpenAI connection
from openai import OpenAI
client = OpenAI(api_key="your-key")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "test"}]
)
```

### Issue 4: Slow Performance

**Solutions**:
1. Use smaller model
2. Enable quantization
3. Reduce `MODEL_CONTEXT_WINDOW`
4. Use GPU with Flash Attention:
   ```python
   ATTN_IMPLEMENTATION = "flash_attention_2"
   ```

## Understanding the Output

### Memory Addition
```
[SUCCESS] Memory added successfully.
```
Memory is stored in `kv_data/` directory as `.pt` files.

### Memory Query
```
The memory stored a period of time ago: [Old memories]

The memory stored just now: [Recent memories]
```
System queries both inactive (old) and active (recent) memory agents.

### Block Full
```
Block {uuid} is full (28800/28800 tokens)
```
Current memory block reached capacity. New block will be created automatically.

## Advanced Configuration

### Custom System Prompts

```python
from src.utils.prompt import CHAT_SYS_PROMPT, ROUTER_SYS_PROMPT

# Customize chat behavior
CUSTOM_CHAT_PROMPT = """
You are a personal assistant with perfect memory.
Always be concise and friendly.
"""

chat = ChatManager(
    model_id="your-model",
    openai_config={"api_key": "key"},
    system_prompt=CUSTOM_CHAT_PROMPT
)
```

### Adjust Memory Block Size

```python
# Larger blocks = fewer blocks, more context per query
MODEL_CONTEXT_WINDOW = 65536  # 64K tokens

# Smaller blocks = more blocks, faster queries
MODEL_CONTEXT_WINDOW = 16384  # 16K tokens
```

### Multiple Tool Rounds

```python
# Allow more complex tool interactions
response = chat.chat(
    user_input="...",
    max_tool_rounds=10  # Default is 5
)
```

## Testing

### Test Memory Persistence

```python
# Session 1
chat = ChatManager(...)
chat.chat("My favorite color is blue.")
# Exit

# Session 2 (new instance)
chat = ChatManager(clean_cache_first=False)  # Don't clear cache
response = chat.chat("What's my favorite color?")
# Should remember: blue
```

### Test Parallel Queries

```python
# Add multiple memories
for i in range(10):
    chat.chat(f"Memory {i}: Some information here.")

# Query should search all blocks in parallel
response = chat.chat("Summarize all my memories.")
```

## Next Steps

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design details
2. Explore [src/utils/prompt.py](src/utils/prompt.py) to customize prompts
3. Check [src/memory/](src/memory/) for core implementation
4. Experiment with different models and configurations

## Getting Help

- Check [README.md](README.md) for overview
- Review [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
- Open an issue on GitHub for bugs
- Read the code comments for implementation details

## Performance Benchmarks

Approximate performance on RTX 3090 (24GB):

| Model Size | Quantization | Memory Usage | Speed (tokens/s) |
|------------|--------------|--------------|------------------|
| 0.5B       | None         | ~2GB         | ~100             |
| 0.5B       | 4-bit        | ~1GB         | ~80              |
| 3B         | None         | ~8GB         | ~40              |
| 3B         | 4-bit        | ~3GB         | ~30              |
| 7B         | 4-bit        | ~6GB         | ~15              |

Note: Speed varies based on context length and hardware.
