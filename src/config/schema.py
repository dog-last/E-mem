"""Configuration schema validation using Pydantic."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""

    api_key: str = Field(..., description="OpenAI API key")
    base_url: str = Field(
        default="https://api.openai.com/v1", description="OpenAI API base URL"
    )
    model: str = Field(default="gpt-4o-mini", description="Model name to use")


class ModelConfig(BaseModel):
    """Model configuration."""

    model_id: str = Field(..., description="HuggingFace model ID or local path")
    openai_config: OpenAIConfig = Field(..., description="OpenAI API configuration")
    model_context_window: int = Field(
        default=32768, ge=1024, le=131072, description="Model context window size"
    )
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] = Field(
        default="sdpa", description="Attention implementation"
    )
    device_map: str = Field(default="auto", description="Device mapping strategy")
    quantization_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Quantization configuration"
    )


class MemoryConfig(BaseModel):
    """Memory system configuration."""

    storage_mode: Literal["kv_cache", "text"] = Field(
        default="kv_cache", description="Storage backend mode"
    )
    clean_cache_first: bool = Field(
        default=True, description="Whether to clean cache on initialization"
    )
    router_system_prompt: Optional[str] = Field(
        default=None, description="Custom router system prompt"
    )
    overlap_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Overlap ratio between KV blocks (0.0-1.0)",
    )
    overlap_mode: Literal["chunk", "token"] = Field(
        default="chunk", description="Overlap handling strategy"
    )
    block_size_ratio: float = Field(
        default=0.125,
        gt=0.0,
        le=1.0,
        description="Block size ratio relative to model context window (0.0-1.0)",
    )
    max_concurrent_gpu_operations: int = Field(
        default=2, ge=1, le=8, description="Number of concurrent GPU operations"
    )
    max_memory_segments: int = Field(
        default=5, ge=1, description="Maximum number of memory segments to return"
    )
    max_blocks: int = Field(
        default=5, ge=1, description="Maximum number of memory blocks for router"
    )
    # Resource control parameters for different hardware configurations
    query_batch_size: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of queries to batch together for inference. "
        "Higher values improve throughput but require more GPU memory. "
        "Set to 2 for single GPU with limited memory, 4-8 for multi-GPU setups.",
    )
    max_parallel_cache_loads: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Maximum number of KV caches to load to GPU in parallel. "
        "Adjust based on available GPU memory.",
    )
    kv_cache_on_gpu: bool = Field(
        default=False,
        description="Keep inactive KV caches on GPU for faster queries. "
        "Only enable if you have sufficient GPU memory (e.g., multiple A100s).",
    )
    enable_router: bool = Field(
        default=True,
        description="Enable LLM-based router for selecting relevant blocks. "
        "Set to False to query ALL blocks directly (useful for evaluation/debugging).",
    )

    @field_validator("overlap_ratio")
    @classmethod
    def validate_overlap_ratio(cls, v: float) -> float:
        """Validate overlap ratio is reasonable."""
        if v > 0.5:
            raise ValueError(
                "overlap_ratio should not exceed 0.5 to maintain block efficiency"
            )
        return v


class LocomoEvalConfig(BaseModel):
    """LoComo evaluation configuration."""

    dataset_path: str = Field(..., description="Path to dataset JSON file")
    output_dir: str = Field(
        default="evaluation/locomo/results", description="Output directory for results"
    )
    ratio: float = Field(
        default=1.0, gt=0.0, le=1.0, description="Portion of dataset to evaluate"
    )
    specific_questions_path: Optional[str] = Field(
        default=None, description="Path to specific questions JSON file"
    )
    conversation_auto_save: bool = Field(
        default=True, description="Enable auto-save for conversation turns"
    )
    categories: List[int] = Field(
        default=[1, 2, 3, 4, 5], description="Categories to evaluate"
    )

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, v: List[int]) -> List[int]:
        """Validate categories are in valid range."""
        for cat in v:
            if cat < 1 or cat > 5:
                raise ValueError(f"Category {cat} out of valid range [1-5]")
        return v


class HotpotQAEvalConfig(BaseModel):
    """HotpotQA evaluation configuration."""

    dataset_path: str = Field(..., description="Path to dataset JSON file")
    output_dir: str = Field(
        default="evaluation/hotpotqa/results",
        description="Output directory for results",
    )
    ratio: float = Field(
        default=1.0, gt=0.0, le=1.0, description="Portion of dataset to evaluate"
    )
    max_tokens_per_chunk: int = Field(
        default=2048, ge=256, description="Maximum tokens per context chunk"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    log_dir: str = Field(
        default="evaluation/locomo/logs", description="Directory for log files"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )


class MaxMemoryConfig(BaseModel):
    """GPU memory configuration per device."""

    model_config = {"extra": "allow"}  # Allow dynamic GPU device keys


class AppConfig(BaseModel):
    """Main application configuration."""

    model: ModelConfig
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    max_memory: Optional[Dict[str, str]] = Field(
        default=None, description="Max memory per GPU device"
    )
    locomo_eval: Optional[LocomoEvalConfig] = Field(
        default=None, description="LoComo evaluation config"
    )
    hotpotqa_eval: Optional[HotpotQAEvalConfig] = Field(
        default=None, description="HotpotQA evaluation config"
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def validate_config_consistency(self) -> "AppConfig":
        """Validate configuration consistency across sections."""
        # Ensure block_size_ratio + overlap_ratio doesn't exceed reasonable limits
        total_ratio = self.memory.block_size_ratio + self.memory.overlap_ratio
        if total_ratio > 0.9:
            raise ValueError(
                f"block_size_ratio ({self.memory.block_size_ratio}) + "
                f"overlap_ratio ({self.memory.overlap_ratio}) = {total_ratio} "
                "exceeds 0.9, which may cause memory issues"
            )
        return self


def load_and_validate_config(config_dict: Dict[str, Any]) -> AppConfig:
    """
    Load and validate configuration from a dictionary.

    Args:
        config_dict: Configuration dictionary (typically from YAML)

    Returns:
        Validated AppConfig instance

    Raises:
        ValidationError: If configuration is invalid
    """
    return AppConfig.model_validate(config_dict)


def validate_openai_config(config_dict: Dict[str, Any]) -> OpenAIConfig:
    """Validate OpenAI configuration standalone."""
    return OpenAIConfig.model_validate(config_dict)


def validate_memory_config(config_dict: Dict[str, Any]) -> MemoryConfig:
    """Validate memory configuration standalone."""
    return MemoryConfig.model_validate(config_dict)

