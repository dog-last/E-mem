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
    """Clear metadata for current session only."""
    if not os.path.exists(METADATA_FILE):
        return
    
    # Get session_id from environment
    session_id = os.environ.get('EVAL_SESSION_ID', 'default')
    
    # Load all metadata
    all_metadata = load_agents_metadata()
    
    # Keep only metadata from other sessions
    other_sessions_metadata = [m for m in all_metadata if m.get("session_id") != session_id]
    
    # Save back
    save_agents_metadata(other_sessions_metadata)
    print(f"Cleared metadata for session: {session_id}")
