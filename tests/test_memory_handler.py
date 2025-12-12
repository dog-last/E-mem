"""Tests for MemoryHandler."""
from unittest.mock import Mock, patch

from src.memory.core.loop_handler import AddHandler, MemoryHandler, QueryHandler


class TestAddHandler:
    """Test AddHandler functionality."""
    
    def test_init(self):
        """Test AddHandler initialization."""
        handler = AddHandler(
            model_id="test-model",
            model_context_window=32768,
            attn_implementation="sdpa",
            device_map="cpu"
        )
        
        assert handler.model_id == "test-model"
        assert handler.active_memory_agent is None
    
    @patch("src.memory.core.loop_handler.MemoryAgent")
    def test_create_agent(self, mock_agent_class):
        """Test creating agent."""
        handler = AddHandler(model_id="test-model")
        handler.create_agent()
        
        assert handler.active_memory_agent is not None
        mock_agent_class.assert_called_once()
    
    @patch("src.memory.core.loop_handler.MemoryAgent")
    def test_add_memory(self, mock_agent_class):
        """Test adding memory."""
        mock_agent = Mock()
        mock_agent.is_active = True
        mock_agent.block_size = 1000
        mock_agent_class.return_value = mock_agent
        
        handler = AddHandler(model_id="test-model")
        is_active = handler.add_memory("Test memory")
        
        assert is_active
        mock_agent.add.assert_called_once_with(["Test memory"])
    
    @patch("src.memory.core.loop_handler.MemoryAgent")
    def test_query_new_agent(self, mock_agent_class):
        """Test querying new agent."""
        mock_agent = Mock()
        mock_agent.query.return_value = "Test response"
        mock_agent_class.return_value = mock_agent
        
        handler = AddHandler(model_id="test-model")
        handler.create_agent()
        result = handler.query_new_agent("Test query")
        
        assert result == "Test response"
        mock_agent.query.assert_called_once_with("Test query")
    
    def test_query_no_agent(self):
        """Test querying when no agent exists."""
        handler = AddHandler(model_id="test-model")
        result = handler.query_new_agent("Test query")
        
        assert result == "No active memory."


class TestQueryHandler:
    """Test QueryHandler functionality."""
    
    def test_init(self, mock_openai_config):
        """Test QueryHandler initialization."""
        with patch("src.memory.core.loop_handler.Router"):
            mock_router = Mock()
            handler = QueryHandler(router=mock_router)
            
            assert handler.router == mock_router
            assert handler.inactive_memory_agent == []
    
    def test_query_memory_no_results(self):
        """Test querying with no results."""
        mock_router = Mock()
        mock_router.map_reduce_blocks.return_value = []
        
        handler = QueryHandler(router=mock_router)
        result = handler.query_memory("Test query")
        
        assert result == "No relevant memory found."
    
    def test_query_memory_with_results(self):
        """Test querying with results."""
        mock_router = Mock()
        mock_router.map_reduce_blocks.return_value = ["Result 1", "Result 2"]
        
        handler = QueryHandler(router=mock_router)
        result = handler.query_memory("Test query")
        
        assert result == "Old Memory Block 1: Result 1\nOld Memory Block 2: Result 2"


class TestMemoryHandler:
    """Test MemoryHandler functionality."""
    
    @patch("src.memory.core.loop_handler.clear_metadata")
    @patch("src.memory.core.loop_handler.clear_kv_cache")
    @patch("src.memory.core.loop_handler.Router")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_init(self, mock_add_handler, mock_router, mock_clear_kv, mock_clear_meta, mock_openai_config):
        """Test MemoryHandler initialization."""
        handler = MemoryHandler(
            model_id="test-model",
            openai_config=mock_openai_config,
            clean_cache_first=True
        )
        
        assert handler.add_handler is not None
        assert handler.inactive_memory_agents == []
        mock_clear_kv.assert_called_once()
        mock_clear_meta.assert_called_once()
    
    @patch("src.memory.core.loop_handler.Router")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_add_memory_active(self, mock_add_handler_class, mock_router):
        """Test adding memory when agent stays active."""
        mock_add_handler = Mock()
        mock_add_handler.add_memory.return_value = True
        mock_add_handler_class.return_value = mock_add_handler
        
        handler = MemoryHandler(model_id="test-model", openai_config={"api_key": "test"})
        handler.add_memory("Test memory")
        
        mock_add_handler.add_memory.assert_called_once_with("Test memory")
        assert len(handler.inactive_memory_agents) == 0
    
    @patch("src.memory.core.loop_handler.save_agents_metadata")
    @patch("src.memory.core.loop_handler.Router")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_add_memory_becomes_inactive(self, mock_add_handler_class, mock_router_class, mock_save_meta):
        """Test adding memory when agent becomes inactive."""
        mock_block = Mock()
        mock_block.block_id = "test-id"
        mock_block.create_timestamp = "20231201_120000"
        mock_block.block_used = 100
        mock_block.chunk_num = 5
        
        mock_agent = Mock()
        mock_agent.is_active = False
        mock_agent.summary = "Test summary"
        mock_agent.current_block = mock_block
        
        mock_add_handler = Mock()
        mock_add_handler.add_memory.return_value = False
        mock_add_handler.active_memory_agent = mock_agent
        mock_add_handler.get_overlap_memories.return_value = []
        mock_add_handler_class.return_value = mock_add_handler
        
        mock_router = Mock()
        mock_router_class.return_value = mock_router
        
        handler = MemoryHandler(model_id="test-model", openai_config={"api_key": "test"})
        handler.add_memory("Test memory")
        
        assert len(handler.inactive_memory_agents) == 1
        mock_add_handler.create_agent.assert_called_once()
        mock_router.add_blocks.assert_called_once_with(mock_agent)
    
    @patch("src.memory.core.loop_handler.Router")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_query_memory_both_sources(self, mock_add_handler_class, mock_router_class):
        """Test querying memory from both old and new sources."""
        mock_add_handler = Mock()
        mock_add_handler.query_new_agent.return_value = "New memory"
        mock_add_handler_class.return_value = mock_add_handler
        
        mock_router = Mock()
        mock_router.map_reduce_blocks.return_value = ["Old memory"]
        mock_router_class.return_value = mock_router
        
        mock_query_handler = Mock()
        mock_query_handler.query_memory.return_value = "Old memory"
        
        handler = MemoryHandler(model_id="test-model", openai_config={"api_key": "test"})
        handler.query_handler.query_memory = mock_query_handler.query_memory
        
        result = handler.query_memory("Test query")
        
        assert "Old memory" in result
        assert "New memory" in result
    
    @patch("src.memory.core.loop_handler.Router")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_query_memory_no_memory(self, mock_add_handler_class, mock_router_class):
        """Test querying when no memory exists."""
        mock_add_handler = Mock()
        mock_add_handler.query_new_agent.return_value = "No active memory."
        mock_add_handler_class.return_value = mock_add_handler
        
        mock_router = Mock()
        mock_router_class.return_value = mock_router
        
        mock_query_handler = Mock()
        mock_query_handler.query_memory.return_value = "No relevant memory found."
        
        handler = MemoryHandler(model_id="test-model", openai_config={"api_key": "test"})
        handler.query_handler.query_memory = mock_query_handler.query_memory
        
        result = handler.query_memory("Test query")
        
        assert result == "No memory found."
    
    @patch("src.memory.core.loop_handler.Router")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_query_memory_only_old(self, mock_add_handler_class, mock_router_class):
        """Test querying with only old memory."""
        mock_add_handler = Mock()
        mock_add_handler.query_new_agent.return_value = "No active memory."
        mock_add_handler_class.return_value = mock_add_handler
        
        mock_router = Mock()
        mock_router_class.return_value = mock_router
        
        mock_query_handler = Mock()
        mock_query_handler.query_memory.return_value = "Old memory content"
        
        handler = MemoryHandler(model_id="test-model", openai_config={"api_key": "test"})
        handler.query_handler.query_memory = mock_query_handler.query_memory
        
        result = handler.query_memory("Test query")
        
        assert result == "Old memory content"
    
    @patch("src.memory.core.loop_handler.Router")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_query_memory_only_new(self, mock_add_handler_class, mock_router_class):
        """Test querying with only new memory."""
        mock_add_handler = Mock()
        mock_add_handler.query_new_agent.return_value = "New memory content"
        mock_add_handler_class.return_value = mock_add_handler
        
        mock_router = Mock()
        mock_router_class.return_value = mock_router
        
        mock_query_handler = Mock()
        mock_query_handler.query_memory.return_value = "No relevant memory found."
        
        handler = MemoryHandler(model_id="test-model", openai_config={"api_key": "test"})
        handler.query_handler.query_memory = mock_query_handler.query_memory
        
        result = handler.query_memory("Test query")
        
        assert result == "New memory content"
    
    @patch("src.memory.core.loop_handler.clear_metadata")
    @patch("src.memory.core.loop_handler.clear_kv_cache")
    @patch("src.memory.core.loop_handler.Router")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_init_no_clean_cache(self, mock_add_handler, mock_router, mock_clear_kv, mock_clear_meta):
        """Test initialization without cleaning cache."""
        _ = MemoryHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
            clean_cache_first=False
        )
        
        mock_clear_kv.assert_not_called()
        mock_clear_meta.assert_not_called()
    
    @patch("src.memory.core.loop_handler.Router")
    @patch("src.memory.core.loop_handler.AddHandler")
    def test_init_with_custom_router_prompt(self, mock_add_handler, mock_router_class):
        """Test initialization with custom router prompt."""
        _ = MemoryHandler(
            model_id="test-model",
            openai_config={"api_key": "test"},
            router_system_prompt="Custom router prompt"
        )
        
        # Verify router was called with custom prompt
        call_args = mock_router_class.call_args
        assert call_args.kwargs["system_prompt"] == "Custom router prompt"
