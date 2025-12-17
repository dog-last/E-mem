"""Tests for ChatManager."""
from unittest.mock import Mock, patch

from src.conversation_manager.chat_handler import ChatManager


class TestChatManager:
    """Test ChatManager functionality."""
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_init(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test ChatManager initialization."""
        chat = ChatManager(
            model_id="test-model",
            openai_config=mock_openai_config
        )
        
        assert chat.name == "chat_manager"
        assert chat.ADD_MEMORY_TOOL["type"] == "function"
        assert chat.SEARCH_MEMORY_TOOL["type"] == "function"
        mock_memory_handler.assert_called_once()
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_chat(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test chat method."""
        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_message.tool_calls = None
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        chat = ChatManager(
            model_id="test-model",
            openai_config=mock_openai_config
        )
        
        response = chat.chat("Hello")
        
        assert response == "Test response"
        assert chat.handle_user_input == "Hello"
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_add_memory_success(self, mock_openai, mock_memory_handler_class, mock_openai_config):
        """Test adding memory successfully."""
        mock_memory_handler = Mock()
        mock_memory_handler.add_memory.return_value = None
        mock_memory_handler_class.return_value = mock_memory_handler
        
        chat = ChatManager(
            model_id="test-model",
            openai_config=mock_openai_config
        )
        chat.handle_user_input = "Test input"
        
        result = chat.add_memory("Test memory")
        
        assert "[SUCCESS]" in result
        mock_memory_handler.add_memory.assert_called_once_with("Test memory")
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_add_memory_error(self, mock_openai, mock_memory_handler_class, mock_openai_config):
        """Test adding memory with error."""
        mock_memory_handler = Mock()
        mock_memory_handler.add_memory.side_effect = Exception("Test error")
        mock_memory_handler_class.return_value = mock_memory_handler
        
        chat = ChatManager(
            model_id="test-model",
            openai_config=mock_openai_config
        )
        
        result = chat.add_memory("Test memory")
        
        assert "[ERROR]" in result
        assert "Test error" in result
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_search_memory_success(self, mock_openai, mock_memory_handler_class, mock_openai_config):
        """Test searching memory successfully."""
        mock_memory_handler = Mock()
        mock_memory_handler.query_memory.return_value = "Found memory"
        mock_memory_handler_class.return_value = mock_memory_handler
        
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        chat = ChatManager(
            model_id="test-model",
            openai_config=mock_openai_config
        )
        
        result = chat.search_memory("Test query")
        
        assert result == "Found memory"
        mock_memory_handler.query_memory.assert_called_once_with("Test query")
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_search_memory_error(self, mock_openai, mock_memory_handler_class, mock_openai_config):
        """Test searching memory with error."""
        mock_memory_handler = Mock()
        mock_memory_handler.query_memory.side_effect = Exception("Query error")
        mock_memory_handler_class.return_value = mock_memory_handler
        
        chat = ChatManager(
            model_id="test-model",
            openai_config=mock_openai_config
        )
        
        result = chat.search_memory("Test query")
        
        assert "[ERROR]" in result
        assert "Query error" in result
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_execute_tool_add_memory(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test executing add_memory tool."""
        chat = ChatManager(
            model_id="test-model",
            openai_config=mock_openai_config
        )
        chat.handle_user_input = "Test input"
        
        with patch.object(chat, "add_memory", return_value="[SUCCESS]") as mock_add:
            result = chat.execute_tool("add_memory", {"memory": "Test"})
            
            assert result == "[SUCCESS]"
            mock_add.assert_called_once_with("Test")
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_execute_tool_query_memory(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test executing query_memory tool."""
        chat = ChatManager(
            model_id="test-model",
            openai_config=mock_openai_config
        )
        
        with patch.object(chat, "search_memory", return_value="Found") as mock_search:
            result = chat.execute_tool("query_memory", {"query": "Test"})
            
            assert result == "Found"
            mock_search.assert_called_once_with("Test")
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_execute_tool_unknown(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test executing unknown tool."""
        chat = ChatManager(
            model_id="test-model",
            openai_config=mock_openai_config
        )
        
        result = chat.execute_tool("unknown_tool", {})
        
        assert "[ERROR]" in result
        assert "Unknown tool" in result
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_add_memory_empty(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test adding empty memory."""
        chat = ChatManager(
            model_id="test-model",
            openai_config=mock_openai_config
        )
        chat.handle_user_input = ""
        
        result = chat.add_memory("")
        
        assert "[ERROR]" in result
        assert "No memory content" in result
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_search_memory_empty_query(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test searching with empty query."""
        chat = ChatManager(
            model_id="test-model",
            openai_config=mock_openai_config
        )
        
        result = chat.search_memory("")
        
        assert "[ERROR]" in result
        assert "No query content" in result
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_save_original_input(self, mock_openai, mock_memory_handler_class, mock_openai_config):
        """Test saving original input instead of processed."""
        mock_memory_handler = Mock()
        mock_memory_handler.add_memory.return_value = None
        mock_memory_handler_class.return_value = mock_memory_handler
        
        chat = ChatManager(
            model_id="test-model",
            openai_config=mock_openai_config
        )
        chat.handle_user_input = "Original input"
        chat.save_original_input = True
        
        chat.add_memory("Processed memory")
        
        mock_memory_handler.add_memory.assert_called_once_with("Original input")
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_chat_with_outer_tools(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test chat with additional outer tools."""
        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_message.tool_calls = None
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        chat = ChatManager(
            model_id="test-model",
            openai_config=mock_openai_config
        )
        
        outer_tool = {"type": "function", "function": {"name": "custom_tool"}}
        chat.chat("Hello", outer_tools=[outer_tool])
        
        # Verify tools were passed
        call_args = mock_client.chat.completions.create.call_args
        assert len(call_args.kwargs["tools"]) == 3  # add_mem + search_mem + custom
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_auto_save_enabled(self, mock_openai, mock_memory_handler_class, mock_openai_config):
        """Test auto_save bypasses LLM and directly saves."""
        mock_memory_handler = Mock()
        mock_memory_handler.add_memory.return_value = None
        mock_memory_handler_class.return_value = mock_memory_handler
        
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        chat = ChatManager(
            model_id="test-model",
            openai_config=mock_openai_config
        )
        
        response = chat.chat("Save this information", auto_save=True)
        
        # Verify memory was saved
        mock_memory_handler.add_memory.assert_called_once_with("Save this information")
        
        # Verify LLM was NOT called
        mock_client.chat.completions.create.assert_not_called()
        
        # Verify response is success message
        assert "[SUCCESS]" in response
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_auto_save_with_save_original_input(self, mock_openai, mock_memory_handler_class, mock_openai_config):
        """Test auto_save with save_original_input flag."""
        mock_memory_handler = Mock()
        mock_memory_handler.add_memory.return_value = None
        mock_memory_handler_class.return_value = mock_memory_handler
        
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        chat = ChatManager(
            model_id="test-model",
            openai_config=mock_openai_config
        )
        
        response = chat.chat(
            "Original input",
            auto_save=True,
            save_original_input=True
        )
        
        # Verify original input was saved
        mock_memory_handler.add_memory.assert_called_once_with("Original input")
        
        # Verify LLM was NOT called
        mock_client.chat.completions.create.assert_not_called()
        
        assert "[SUCCESS]" in response
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_auto_save_error_handling(self, mock_openai, mock_memory_handler_class, mock_openai_config):
        """Test auto_save handles errors properly."""
        mock_memory_handler = Mock()
        mock_memory_handler.add_memory.side_effect = Exception("Save failed")
        mock_memory_handler_class.return_value = mock_memory_handler
        
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        chat = ChatManager(
            model_id="test-model",
            openai_config=mock_openai_config
        )
        
        response = chat.chat("Save this", auto_save=True)
        
        # Verify error message is returned
        assert "[ERROR]" in response
        assert "Save failed" in response
        
        # Verify LLM was NOT called
        mock_client.chat.completions.create.assert_not_called()
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_auto_save_disabled_uses_llm(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test that auto_save=False uses normal LLM flow."""
        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "LLM response"
        mock_message.tool_calls = None
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        chat = ChatManager(
            model_id="test-model",
            openai_config=mock_openai_config
        )
        
        response = chat.chat("Hello", auto_save=False)
        
        # Verify LLM was called
        mock_client.chat.completions.create.assert_called_once()
        
        # Verify response is from LLM
        assert response == "LLM response"
