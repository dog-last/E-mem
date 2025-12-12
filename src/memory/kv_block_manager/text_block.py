import json
import os
import uuid
from datetime import datetime

TEXT_DATA_DIR = os.path.join(os.getcwd(), "text_data")
os.makedirs(TEXT_DATA_DIR, exist_ok=True)


class TextBlock:
    def __init__(self, block_id: uuid.UUID, create_timestamp: str, block_size: int = 32000):
        self.block_id = block_id
        self.create_timestamp = create_timestamp
        self.store_target = os.path.join(TEXT_DATA_DIR, f"text_block_{self.block_id}_{self.create_timestamp}.json")
        self.block_size = block_size
        self.block_used = 0
        self.chunk_num = 0
        self.chunks = []
        if not os.path.exists(self.store_target):
            self._save()

    def add_chunk(self, text: str, token_count: int) -> bool:
        self.chunks.append({"text": text, "tokens": token_count, "timestamp": datetime.now().isoformat()})
        self.block_used += token_count
        self.chunk_num += 1
        self._save()
        return self.is_full()

    def get_all_text(self) -> str:
        return "\n\n".join([f"[Context {i+1}]\n{chunk['text']}" for i, chunk in enumerate(self.chunks)])

    def _save(self):
        with open(self.store_target, 'w', encoding='utf-8') as f:
            json.dump({"block_id": str(self.block_id), "create_timestamp": self.create_timestamp,
                      "block_size": self.block_size, "block_used": self.block_used,
                      "chunk_num": self.chunk_num, "chunks": self.chunks}, f, ensure_ascii=False, indent=2)

    def load(self):
        if not os.path.exists(self.store_target):
            return
        with open(self.store_target, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.block_used = data.get("block_used", 0)
            self.chunk_num = data.get("chunk_num", 0)
            self.chunks = data.get("chunks", [])

    def is_full(self) -> bool:
        return self.block_used >= self.block_size


def clear_text_cache():
    if not os.path.exists(TEXT_DATA_DIR):
        return
    for file in os.listdir(TEXT_DATA_DIR):
        if file.endswith(".json") and not file.endswith("agents_metadata.json"):
            os.remove(os.path.join(TEXT_DATA_DIR, file))


# Metadata functions
TEXT_METADATA_FILE = os.path.join(TEXT_DATA_DIR, "agents_metadata.json")

def save_text_agents_metadata(agents_data: list):
    with open(TEXT_METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(agents_data, f, ensure_ascii=False, indent=2)

def load_text_agents_metadata() -> list:
    if not os.path.exists(TEXT_METADATA_FILE):
        return []
    with open(TEXT_METADATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def clear_text_metadata():
    if os.path.exists(TEXT_METADATA_FILE):
        os.remove(TEXT_METADATA_FILE)
