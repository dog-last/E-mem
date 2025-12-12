"""Tests for Router."""
from unittest.mock import Mock, patch

import pytest

from src.memory.router.router import Router


class TestRouter:
    """Test Router functionality."""
    
    @patch("src.agent.base.OpenAI")
    def test_init(self, mock_openai, mock_openai_config):
        """Test Router initialization."""
        router = Router(openai_config=mock_openai_config)
        
        assert router.name == "router"
        assert router.agent == []
        mock_openai.assert_called_once()
    
    def test_init_no_config(self):
        """Test initialization without config raises error."""
        with pytest.raises(NotImplementedError):
            Router(openai_config=None)
    
    @patch("src.agent.base.OpenAI")
    def test_add_blocks_inactive(self, mock_openai, mock_openai_config):
        """Test adding inactive memory agent."""
        router = Router(openai_config=mock_openai_config)
        
        mock_agent = Mock()
        mock_agent.is_active = False
        
        router.add_blocks(mock_agent)
        
        assert len(router.agent) == 1
        assert router.agent[0] == mock_agent
    
    @patch("src.agent.base.OpenAI")
    def test_add_blocks_active_ignored(self, mock_openai, mock_openai_config):
        """Test that active agents are not added."""
        router = Router(openai_config=mock_openai_config)
        
        mock_agent = Mock()
        mock_agent.is_active = True
        
        router.add_blocks(mock_agent)
        
        assert len(router.agent) == 0
    
    @patch("src.agent.base.OpenAI")
    def test_map_blocks_no_agents(self, mock_openai, mock_openai_config):
        """Test mapping with no agents."""
        router = Router(openai_config=mock_openai_config)
        
        result = router._map_blocks("Test query")
        
        assert result == []
    
    @patch("src.agent.base.OpenAI")
    def test_map_blocks_with_agents(self, mock_openai, mock_openai_config):
        """Test mapping with agents."""
        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "<summary_index>0,1</summary_index>"
        mock_message.tool_calls = None
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        router = Router(openai_config=mock_openai_config)
        
        # Add mock agents
        mock_agent1 = Mock()
        mock_agent1.is_active = False
        mock_agent1.summary = "Summary 1"
        
        mock_agent2 = Mock()
        mock_agent2.is_active = False
        mock_agent2.summary = "Summary 2"
        
        router.add_blocks(mock_agent1)
        router.add_blocks(mock_agent2)
        
        result = router._map_blocks("Test query")
        
        assert len(result) == 2
        assert result[0] == mock_agent1
        assert result[1] == mock_agent2
    
    @patch("src.agent.base.OpenAI")
    def test_map_blocks_max_blocks_limit(self, mock_openai, mock_openai_config):
        """Test max_blocks parameter limits results."""
        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "<summary_index>0,1,2</summary_index>"
        mock_message.tool_calls = None
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        router = Router(openai_config=mock_openai_config)
        
        # Add 3 mock agents
        for i in range(3):
            mock_agent = Mock()
            mock_agent.is_active = False
            mock_agent.summary = f"Summary {i}"
            router.add_blocks(mock_agent)
        
        result = router._map_blocks("Test query", max_blocks=2)
        
        assert len(result) == 2
    
    @patch("src.agent.base.OpenAI")
    def test_map_reduce_blocks(self, mock_openai, mock_openai_config):
        """Test map_reduce_blocks."""
        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "<summary_index>0</summary_index>"
        mock_message.tool_calls = None
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        router = Router(openai_config=mock_openai_config)
        
        mock_agent = Mock()
        mock_agent.is_active = False
        mock_agent.summary = "Summary"
        mock_agent.query.return_value = "Agent response"
        
        router.add_blocks(mock_agent)
        
        result = router.map_reduce_blocks("Test query")
        
        assert len(result) == 1
        assert result[0] == "Agent response"
        mock_agent.query.assert_called_once_with("Test query")
    
    @patch("src.agent.base.OpenAI")
    def test_execute_tool(self, mock_openai, mock_openai_config):
        """Test execute_tool does nothing."""
        router = Router(openai_config=mock_openai_config)
        result = router.execute_tool("test", {})
        assert result is None
    
    @patch("src.agent.base.OpenAI")
    def test_map_blocks_invalid_response(self, mock_openai, mock_openai_config):
        """Test mapping with invalid LLM response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "No tags here"
        mock_message.tool_calls = None
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        router = Router(openai_config=mock_openai_config)
        
        mock_agent = Mock()
        mock_agent.is_active = False
        mock_agent.summary = "Summary"
        router.add_blocks(mock_agent)
        
        result = router._map_blocks("Test query", max_blocks=5)
        
        # When no summary_index tag found, returns all blocks up to max_blocks
        assert len(result) == 1
    
    @patch("src.agent.base.OpenAI")
    def test_map_blocks_out_of_range_indices(self, mock_openai, mock_openai_config):
        """Test mapping with out of range indices."""
        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "<summary_index>0,5,10</summary_index>"
        mock_message.tool_calls = None
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        router = Router(openai_config=mock_openai_config)
        
        # Add only 2 agents
        for i in range(2):
            mock_agent = Mock()
            mock_agent.is_active = False
            mock_agent.summary = f"Summary {i}"
            router.add_blocks(mock_agent)
        
        result = router._map_blocks("Test query")
        
        # Should only return valid indices (0)
        assert len(result) == 1
    
    @patch("src.agent.base.OpenAI")
    def test_map_reduce_blocks_empty(self, mock_openai, mock_openai_config):
        """Test map_reduce_blocks with no agents."""
        router = Router(openai_config=mock_openai_config)
        
        result = router.map_reduce_blocks("Test query")
        
        assert result == []
    
    @patch("src.agent.base.OpenAI")
    def test_map_blocks_exception_handling(self, mock_openai, mock_openai_config):
        """Test exception handling in _map_blocks."""
        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "<summary_index>invalid</summary_index>"
        mock_message.tool_calls = None
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        router = Router(openai_config=mock_openai_config)
        
        mock_agent = Mock()
        mock_agent.is_active = False
        mock_agent.summary = "Summary"
        router.add_blocks(mock_agent)
        
        result = router._map_blocks("Test query", max_blocks=5)
        
        # When parsing fails (no valid integers), returns empty list
        assert len(result) == 0
