import json
import os
from typing import Dict, List

METADATA_FILE = os.path.join(os.getcwd(), "kv_data", "agents_metadata.json")


def save_agents_metadata(agents_data: List[Dict]):
    """Save agents metadata to JSON file."""
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(agents_data, f, ensure_ascii=False, indent=2)


def load_agents_metadata() -> List[Dict]:
    """Load agents metadata from JSON file."""
    if not os.path.exists(METADATA_FILE):
        return []
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def clear_metadata():
    """Clear metadata file."""
    if os.path.exists(METADATA_FILE):
        os.remove(METADATA_FILE)
