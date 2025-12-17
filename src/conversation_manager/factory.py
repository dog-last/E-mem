"""Factory for creating ChatManager with different storage backends."""

import logging
from typing import Any, Dict, Literal, Optional

from src.conversation_manager.chat_handler import ChatManager, TextStorageChatManager

logger = logging.getLogger(__name__)


def create_chat_manager(
    storage_mode: Literal["kv_cache", "text"] = "kv_cache",
    model_id: str = "Qwen/Qwen3-4B",
    openai_config: Optional[Dict[str, Any]] = None,
    clean_cache_first: bool = True,
    model_context_window: int = 32768,
    router_system_prompt: Optional[str] = None,
    overlap_mode: str = "chunk",
    overlap_ratio: float = 0.1,
    block_size_ratio: float = 0.125,
    max_memory_segments: Optional[int] = None,
    max_blocks: int = 5,
    query_batch_size: int = 4,
    max_parallel_cache_loads: int = 8,
    enable_router: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Factory function to create ChatManager with specified storage backend.
    
    Uses shared model architecture for memory-efficient multi-block queries.
    Supports both single-GPU and multi-GPU configurations.

    Args:
        storage_mode: "kv_cache" for KV cache storage, "text" for text-based storage
        model_id: Model identifier for HuggingFace
        openai_config: OpenAI API configuration dict
        clean_cache_first: Whether to clear existing cache on initialization
        model_context_window: Context window size for the model
        router_system_prompt: Custom system prompt for router
        overlap_mode: Overlap handling strategy ('chunk' or 'token')
        overlap_ratio: Overlap ratio between blocks (0.0-1.0)
        block_size_ratio: Block size relative to context window (0.0-1.0)
        max_memory_segments: Maximum memory segments to return per block query
        max_blocks: Maximum number of memory blocks to select by router
        query_batch_size: Queries to batch together (higher = more throughput, more memory)
        max_parallel_cache_loads: Max parallel KV cache loads to GPU
        enable_router: If False, skip LLM routing and query ALL blocks directly
        **kwargs: Additional arguments passed to ChatManager
            For kv_cache mode: attn_implementation, device_map, quantization_config, max_memory, offload_folder

    Returns:
        ChatManager or TextStorageChatManager instance

    Example:
        # KV Cache mode with resource optimization for multi-GPU
        chat_manager = create_chat_manager(
            storage_mode="kv_cache",
            model_id="Qwen/Qwen3-4B",
            openai_config={"api_key": "your-key"},
            max_memory_segments=5,
            max_blocks=5,
            max_parallel_cache_loads=8,  # For multi-GPU with lots of memory
        )

        # KV Cache mode for single GPU with limited memory
        chat_manager = create_chat_manager(
            storage_mode="kv_cache",
            model_id="Qwen/Qwen3-4B",
            openai_config={"api_key": "your-key"},
            max_parallel_cache_loads=2,  # Conservative for limited memory
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
            overlap_ratio=overlap_ratio,
            block_size_ratio=block_size_ratio,
            max_memory_segments=max_memory_segments,
            max_blocks=max_blocks,
            query_batch_size=query_batch_size,
            max_parallel_cache_loads=max_parallel_cache_loads,
            enable_router=enable_router,
            **kwargs,
        )
    elif storage_mode == "text":
        # Text storage doesn't need GPU-related parameters
        gpu_params = [
            "attn_implementation",
            "device_map",
            "quantization_config",
            "max_memory",
            "offload_folder",
            "query_batch_size",
            "max_parallel_cache_loads",
        ]
        ignored_params = [k for k in kwargs if k in gpu_params]
        if ignored_params:
            logger.warning(
                f"Ignoring KV cache parameters for text storage: {ignored_params}"
            )

        return TextStorageChatManager(
            model_id=model_id,
            openai_config=openai_config,
            clean_cache_first=clean_cache_first,
            model_context_window=model_context_window,
            router_system_prompt=router_system_prompt,
            overlap_mode=overlap_mode,
            overlap_ratio=overlap_ratio,
            block_size_ratio=block_size_ratio,
            max_memory_segments=max_memory_segments,
            max_blocks=max_blocks,
            enable_router=enable_router,
        )
    else:
        raise ValueError(
            f"Invalid storage_mode: {storage_mode}. Must be 'kv_cache' or 'text'"
        )
