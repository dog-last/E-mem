"""Configuration loader for memory system."""
import os

import yaml

# Default values
MAX_CONCURRENT_GPU_OPERATIONS = 2
DEFAULT_OVERLAP_RATIO = 0.1

# Try to load from config.yaml if exists
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
if os.path.exists(config_path):
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            if config_data and 'memory' in config_data:
                MAX_CONCURRENT_GPU_OPERATIONS = config_data['memory'].get('max_concurrent_gpu_operations', 2)
                DEFAULT_OVERLAP_RATIO = config_data['memory'].get('overlap_ratio', 0.1)
    except Exception:
        pass  # Use defaults if loading fails
