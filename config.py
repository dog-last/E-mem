"""Configuration loader for memory system with Pydantic validation."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

import yaml

if TYPE_CHECKING:
    from src.config import AppConfig

logger = logging.getLogger(__name__)

# Default values
MAX_CONCURRENT_GPU_OPERATIONS: int = 2
DEFAULT_OVERLAP_RATIO: float = 0.1
DEFAULT_BLOCK_SIZE_RATIO: float = 0.125

# Cached config instance
_config_cache: Optional[Dict[str, Any]] = None


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    pass


def get_config_path() -> str:
    """Get the configuration file path."""
    return os.path.join(os.path.dirname(__file__), "config.yaml")


def load_raw_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load raw configuration from YAML file.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        Configuration dictionary.

    Raises:
        ConfigurationError: If config file cannot be loaded.
    """
    if config_path is None:
        config_path = get_config_path()

    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
            return config_data if config_data else {}
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in config file: {e}") from e
    except OSError as e:
        raise ConfigurationError(f"Cannot read config file: {e}") from e


def load_validated_config(config_path: Optional[str] = None) -> "AppConfig":
    """
    Load and validate configuration using Pydantic.

    Args:
        config_path: Optional path to config file.

    Returns:
        Validated AppConfig instance.

    Raises:
        ConfigurationError: If validation fails.
    """
    from pydantic import ValidationError

    from src.config import load_and_validate_config

    raw_config = load_raw_config(config_path)

    try:
        return load_and_validate_config(raw_config)
    except ValidationError as e:
        error_messages = []
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]
            error_messages.append(f"  - {loc}: {msg}")
        raise ConfigurationError(
            "Configuration validation failed:\n" + "\n".join(error_messages)
        ) from e


def get_config(reload: bool = False) -> Dict[str, Any]:
    """
    Get configuration with caching.

    Args:
        reload: Force reload from file.

    Returns:
        Configuration dictionary.
    """
    global _config_cache

    if _config_cache is None or reload:
        _config_cache = load_raw_config()

    return _config_cache


def _update_globals_from_config() -> None:
    """Update global configuration values from config file."""
    global MAX_CONCURRENT_GPU_OPERATIONS, DEFAULT_OVERLAP_RATIO, DEFAULT_BLOCK_SIZE_RATIO

    config_data = get_config()
    if config_data and "memory" in config_data:
        memory_config = config_data["memory"]
        MAX_CONCURRENT_GPU_OPERATIONS = memory_config.get(
            "max_concurrent_gpu_operations", 2
        )
        DEFAULT_OVERLAP_RATIO = memory_config.get("overlap_ratio", 0.1)
        DEFAULT_BLOCK_SIZE_RATIO = memory_config.get("block_size_ratio", 0.125)


# Initialize on module load
_update_globals_from_config()
