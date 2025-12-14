"""Factory for creating ChatManager with different storage backends."""
import logging
from typing import Literal

from src.conversation_manager.chat_handler import ChatManager, TextStorageChatManager

logger = logging.getLogger(__name__)


def create_chat_manager(
    storage_mode: Literal["kv_cache", "text"] = "kv_cache",
    model_id: str = "Qwen/Qwen3-4B",
    openai_config: dict = None,
    clean_cache_first: bool = True,
    model_context_window: int = 32768,
    router_system_prompt: str = None,
    overlap_mode: str = "chunk",
    **kwargs
):
    """
    Factory function to create ChatManager with specified storage backend.
    
    Args:
        storage_mode: "kv_cache" for KV cache storage, "text" for text-based storage
        model_id: Model identifier for HuggingFace
        openai_config: OpenAI API configuration dict
        clean_cache_first: Whether to clear existing cache on initialization
        model_context_window: Context window size for the model
        router_system_prompt: Custom system prompt for router
        **kwargs: Additional arguments passed to ChatManager
            For kv_cache mode: attn_implementation, device_map, quantization_config, max_memory, offload_folder
    
    Returns:
        ChatManager or TextStorageChatManager instance
    
    Example:
        # KV Cache mode (default)
        chat_manager = create_chat_manager(
            storage_mode="kv_cache",
            model_id="Qwen/Qwen3-4B",
            openai_config={"api_key": "your-key"}
        )
        
        # Text storage mode
        chat_manager = create_chat_manager(
            storage_mode="text",
            model_id="Qwen/Qwen3-4B",
            openai_config={"api_key": "your-key"}
        )
    """
    logger.info(f"Creating ChatManager with storage_mode={storage_mode}")
    
    if storage_mode == "kv_cache":
        return ChatManager(
            model_id=model_id,
            openai_config=openai_config,
            clean_cache_first=clean_cache_first,
            model_context_window=model_context_window,
            router_system_prompt=router_system_prompt,
            overlap_mode=overlap_mode,
            **kwargs
        )
    elif storage_mode == "text":
        # Text storage doesn't need GPU-related parameters
        if kwargs:
            logger.warning(f"Ignoring KV cache parameters for text storage: {list(kwargs.keys())}")
        return TextStorageChatManager(
            model_id=model_id,
            openai_config=openai_config,
            clean_cache_first=clean_cache_first,
            model_context_window=model_context_window,
            router_system_prompt=router_system_prompt,
            overlap_mode=overlap_mode
        )
    else:
        raise ValueError(f"Invalid storage_mode: {storage_mode}. Must be 'kv_cache' or 'text'")
