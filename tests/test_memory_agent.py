"""Tests for MemoryAgent."""
from unittest.mock import Mock, patch

import pytest
import torch

from src.memory.memory_agent.agent import MemoryAgent


class TestMemoryAgent:
    """Test MemoryAgent functionality."""
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_init(self, mock_block, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test MemoryAgent initialization."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n<|im_start|>user\nTEST<|im_end|>\n"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.device = torch.device("cpu")
        del mock_model.hf_device_map
        mock_model_class.from_pretrained.return_value = mock_model
        
        agent = MemoryAgent(model_id="test-model", model_context_window=1000)
        
        assert agent.model_id == "test-model"
        assert agent.block_size == 900
        assert agent.is_active
        assert agent.global_offset == 0
        assert agent.chunk_number == 0
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_extract_chat_tokens(self, mock_block, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test chat token extraction."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>user\nTEST<|im_end|>\n"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.device = torch.device("cpu")
        del mock_model.hf_device_map
        mock_model_class.from_pretrained.return_value = mock_model
        
        agent = MemoryAgent(model_id="test-model")
        
        assert agent.role_start == "<|im_start|>"
        assert agent.role_end == "<|im_end|>"
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_add_knowledge_first_chunk(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test adding first knowledge chunk."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.device = torch.device("cpu")
        del mock_model.hf_device_map
        mock_output = Mock()
        mock_output.past_key_values = [(torch.randn(1, 8, 5, 64), torch.randn(1, 8, 5, 64)) for _ in range(4)]
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_block = Mock()
        mock_block.save_cache.return_value = False
        mock_block_class.return_value = mock_block
        
        agent = MemoryAgent(model_id="test-model", model_context_window=1000)
        result = agent._add_knowledge(["Test chunk"])
        
        assert not result
        assert agent.chunk_number == 1
        assert agent.global_offset == 5
        assert len(agent.saved_chunks) == 1
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_add_knowledge_block_full(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test adding knowledge when block becomes full."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.device = torch.device("cpu")
        del mock_model.hf_device_map
        mock_output = Mock()
        mock_output.past_key_values = [(torch.randn(1, 8, 3, 64), torch.randn(1, 8, 3, 64)) for _ in range(4)]
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_block = Mock()
        mock_block.save_cache.return_value = True
        mock_block_class.return_value = mock_block
        
        agent = MemoryAgent(model_id="test-model", model_context_window=100)
        result = agent._add_knowledge(["Test chunk"])
        
        assert result
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_add_active_agent(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test add method on active agent."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.device = torch.device("cpu")
        del mock_model.hf_device_map
        mock_output = Mock()
        mock_output.past_key_values = [(torch.randn(1, 8, 3, 64), torch.randn(1, 8, 3, 64)) for _ in range(4)]
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_block = Mock()
        mock_block.save_cache.return_value = False
        mock_block_class.return_value = mock_block
        
        agent = MemoryAgent(model_id="test-model")
        agent.add(["Test chunk"])
        
        assert agent.is_active
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_add_becomes_inactive(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test add method when agent becomes inactive."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "Summary text"
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.device = torch.device("cpu")
        del mock_model.hf_device_map
        mock_output = Mock()
        mock_output.past_key_values = [(torch.randn(1, 8, 3, 64), torch.randn(1, 8, 3, 64)) for _ in range(4)]
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_block = Mock()
        mock_block.save_cache.return_value = True
        mock_block_class.return_value = mock_block
        
        agent = MemoryAgent(model_id="test-model", model_context_window=100)
        agent.add(["Test chunk"])
        
        assert not agent.is_active
        assert agent.summary is not None
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_query_no_chunks(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test querying with no saved chunks."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.device = torch.device("cpu")
        del mock_model.hf_device_map
        mock_model_class.from_pretrained.return_value = mock_model
        
        agent = MemoryAgent(model_id="test-model")
        result = agent.query("Test question")
        
        assert result == "No knowledge available."
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_query_with_chunks(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test querying with saved chunks."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "Answer text"
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.device = torch.device("cpu")
        del mock_model.hf_device_map
        mock_output = Mock()
        mock_output.past_key_values = [(torch.randn(1, 8, 3, 64), torch.randn(1, 8, 3, 64)) for _ in range(4)]
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_block = Mock()
        mock_block.save_cache.return_value = False
        mock_block_class.return_value = mock_block
        
        agent = MemoryAgent(model_id="test-model")
        agent.saved_chunks = [{
            "cache": [(torch.randn(1, 8, 5, 64), torch.randn(1, 8, 5, 64)) for _ in range(4)],
            "start": 0,
            "length": 5
        }]
        agent.global_offset = 5
        
        result = agent.query("Test question")
        
        assert result == "Answer text"
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_remove_thinking_content(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test removing thinking tags from response."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.device = torch.device("cpu")
        del mock_model.hf_device_map
        mock_model_class.from_pretrained.return_value = mock_model
        
        agent = MemoryAgent(model_id="test-model")
        
        text_with_thinking = "Answer <thinking>internal thoughts</thinking> more text"
        cleaned = agent._remove_thinking_content(text_with_thinking)
        
        assert "<thinking>" not in cleaned
        assert "internal thoughts" not in cleaned
        assert "Answer" in cleaned
        assert "more text" in cleaned
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_agent_generate_no_question_or_instruction(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test _agent_generate raises error without question or instruction."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.device = torch.device("cpu")
        del mock_model.hf_device_map
        mock_model_class.from_pretrained.return_value = mock_model
        
        agent = MemoryAgent(model_id="test-model")
        agent.saved_chunks = [{
            "cache": [(torch.randn(1, 8, 5, 64), torch.randn(1, 8, 5, 64)) for _ in range(4)],
            "start": 0,
            "length": 5
        }]
        
        with pytest.raises(ValueError, match="Either question or instruction"):
            agent._agent_generate()
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_add_multiple_chunks(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test adding multiple chunks."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.device = torch.device("cpu")
        del mock_model.hf_device_map
        mock_output = Mock()
        mock_output.past_key_values = [(torch.randn(1, 8, 3, 64), torch.randn(1, 8, 3, 64)) for _ in range(4)]
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_block = Mock()
        mock_block.save_cache.return_value = False
        mock_block_class.return_value = mock_block
        
        agent = MemoryAgent(model_id="test-model")
        agent._add_knowledge(["Chunk 1", "Chunk 2", "Chunk 3"])
        
        assert agent.chunk_number == 3
        assert len(agent.saved_chunks) == 3
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_multi_gpu_device_handling(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test device handling for multi-GPU setup."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Simulate multi-GPU model with hf_device_map
        mock_model = Mock()
        mock_model.hf_device_map = {"model.layers.0": "cuda:0", "model.layers.1": "cuda:1"}
        mock_model.device = torch.device("cuda:0")
        mock_model_class.from_pretrained.return_value = mock_model
        
        agent = MemoryAgent(model_id="test-model")
        
        # Verify primary_device is set correctly
        assert hasattr(agent, "primary_device")
        assert isinstance(agent.primary_device, torch.device)
    
    @patch("src.memory.memory_agent.agent.AutoModelForCausalLM")
    @patch("src.memory.memory_agent.agent.AutoTokenizer")
    @patch("src.memory.memory_agent.agent.KVBlock")
    def test_single_gpu_device_handling(self, mock_block_class, mock_tokenizer_class, mock_model_class, temp_kv_dir):
        """Test device handling for single GPU setup."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Simulate single GPU model without hf_device_map
        mock_model = Mock()
        mock_model.device = torch.device("cuda:0")
        del mock_model.hf_device_map
        mock_model_class.from_pretrained.return_value = mock_model
        
        agent = MemoryAgent(model_id="test-model")
        
        # Verify primary_device is set to model.device
        assert agent.primary_device == mock_model.device
