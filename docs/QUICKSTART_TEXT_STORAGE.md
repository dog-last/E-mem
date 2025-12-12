# Quick Start: Text Storage Mode

## 30秒快速开始

```python
from src.conversation_manager.factory import create_chat_manager

# 创建Text Storage模式的ChatManager
manager = create_chat_manager(
    storage_mode="text",  # 选择text模式
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    openai_config={
        "api_key": "your-openai-api-key",
        "base_url": "https://api.openai.com/v1"  # 可选
    }
)

# 添加记忆
manager.chat("My name is Alice and I love Python.", auto_save=True)

# 查询记忆
response = manager.chat("What is my name?")
print(response)
```

## 切换到KV Cache模式

只需修改一个参数：

```python
manager = create_chat_manager(
    storage_mode="kv_cache",  # 改为kv_cache
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    openai_config={"api_key": "your-key"}
)
```

## 完整示例

```python
from src.conversation_manager.factory import create_chat_manager

# 初始化
manager = create_chat_manager(
    storage_mode="text",
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    openai_config={"api_key": "sk-xxx"},
    clean_cache_first=True,  # 清空旧缓存
    model_context_window=32768
)

# 场景1：自动保存模式
manager.chat("I work at Google as a software engineer.", auto_save=True)
manager.chat("My favorite programming language is Rust.", auto_save=True)

# 场景2：让LLM决定是否保存
manager.chat("Remember that my birthday is on March 15th.")

# 场景3：查询记忆
response = manager.chat("Where do I work?")
print(response)

response = manager.chat("What's my favorite language?")
print(response)
```

## 配置说明

### 必需参数
- `storage_mode`: `"text"` 或 `"kv_cache"`
- `model_id`: HuggingFace模型ID（用于tokenizer）
- `openai_config`: OpenAI API配置

### 可选参数
- `clean_cache_first`: 是否清空缓存（默认True）
- `model_context_window`: 上下文窗口大小（默认32768）
- `router_system_prompt`: 自定义Router提示词

## 存储位置

- Text模式：`./text_data/*.json`
- KV Cache模式：`./kv_data/*.pt`

## 运行示例

```bash
# 运行完整示例
python3 example_text_storage.py

# 运行测试
python3 test_text_storage.py
```

## 常见问题

**Q: 两种模式可以同时使用吗？**  
A: 可以，它们使用不同的存储目录，互不干扰。

**Q: 如何清空缓存？**  
A: 设置 `clean_cache_first=True` 或手动删除 `text_data/` 目录。

**Q: Text模式需要GPU吗？**  
A: 不需要，仅需要OpenAI API密钥。

**Q: 哪种模式更快？**  
A: KV Cache模式更快，但需要GPU。Text模式适合无GPU环境。

## 下一步

- 查看 `TEXT_STORAGE_README.md` 了解详细架构
- 查看 `IMPLEMENTATION_SUMMARY.md` 了解实现细节
- 运行 `example_text_storage.py` 查看完整示例
