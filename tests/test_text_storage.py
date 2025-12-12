"""Test text storage implementation."""
import uuid
from datetime import datetime

import pytest

from src.conversation_manager.chat_handler import TextStorageChatManager
from src.conversation_manager.factory import create_chat_manager
from src.memory.core.text_loop_handler import TextMemoryHandler
from src.memory.kv_block_manager.text_block import TextBlock, clear_text_cache
from src.memory.memory_agent.text_agent import TextMemoryAgent


def test_imports():
    """Test that all modules can be imported."""
    assert TextBlock is not None
    assert TextMemoryAgent is not None
    assert TextMemoryHandler is not None
    assert create_chat_manager is not None
    assert TextStorageChatManager is not None


def test_text_block():
    """Test TextBlock functionality."""
    block = TextBlock(
        block_id=uuid.uuid4(),
        create_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        block_size=1000
    )
    
    # Add chunk
    is_full = block.add_chunk("Test memory chunk", 100)
    assert block.chunk_num == 1
    assert block.block_used == 100
    assert not is_full
    
    # Get text
    text = block.get_all_text()
    assert "Test memory chunk" in text
    
    # Test full detection
    block.add_chunk("Large memory", 900)
    assert block.is_full()
    
    # Cleanup
    clear_text_cache()


def test_factory_function():
    """Test factory function validation."""
    with pytest.raises(ValueError, match="Invalid storage_mode"):
        create_chat_manager(storage_mode="invalid")


def test_cleanup():
    """Test cache cleanup."""
    clear_text_cache()
    # Should not raise any errors
