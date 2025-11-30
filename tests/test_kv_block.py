"""Tests for KVBlock."""
import os

from src.memory.kv_block_manager.block import KVBlock, clear_cache


class TestKVBlock:
    """Test KVBlock functionality."""
    
    def test_init(self, temp_kv_dir, block_id, timestamp):
        """Test KVBlock initialization."""
        block = KVBlock(block_id=block_id, create_timestamp=timestamp, block_size=1000)
        
        assert block.block_id == block_id
        assert block.create_timestamp == timestamp
        assert block.block_size == 1000
        assert block.block_used == 0
        assert block.chunk_num == 0
        assert os.path.exists(block.store_target)
    
    def test_save_cache(self, temp_kv_dir, block_id, timestamp, sample_cache_state):
        """Test saving cache."""
        block = KVBlock(block_id=block_id, create_timestamp=timestamp, block_size=1000)
        
        is_full = block.save_cache(sample_cache_state, 100)
        
        assert block.block_used == 100
        assert block.chunk_num == 2
        assert not is_full
    
    def test_save_cache_full(self, temp_kv_dir, block_id, timestamp, sample_cache_state):
        """Test saving cache when block becomes full."""
        block = KVBlock(block_id=block_id, create_timestamp=timestamp, block_size=100)
        
        is_full = block.save_cache(sample_cache_state, 100)
        
        assert is_full
        assert block.block_used == 100
    
    def test_load_cache(self, temp_kv_dir, block_id, timestamp, sample_cache_state):
        """Test loading cache."""
        block = KVBlock(block_id=block_id, create_timestamp=timestamp, block_size=1000)
        block.save_cache(sample_cache_state, 100)
        
        loaded = block.load_cache()
        
        assert loaded["global_offset"] == 100
        assert loaded["chunk_number"] == 2
        assert loaded["model_id"] == "test-model"
    
    def test_is_full(self, temp_kv_dir, block_id, timestamp):
        """Test is_full method."""
        block = KVBlock(block_id=block_id, create_timestamp=timestamp, block_size=100)
        
        assert not block.is_full()
        
        block.block_used = 100
        assert block.is_full()
        
        block.block_used = 101
        assert block.is_full()
    
    def test_clear_cache(self, temp_kv_dir, block_id, timestamp, sample_cache_state):
        """Test clearing cache."""
        block1 = KVBlock(block_id=block_id, create_timestamp=timestamp, block_size=1000)
        block1.save_cache(sample_cache_state, 100)
        
        block2 = KVBlock(block_id=block_id, create_timestamp=timestamp + "2", block_size=1000)
        block2.save_cache(sample_cache_state, 100)
        
        assert os.path.exists(block1.store_target)
        assert os.path.exists(block2.store_target)
        
        clear_cache()
        
        assert not os.path.exists(block1.store_target)
        assert not os.path.exists(block2.store_target)
