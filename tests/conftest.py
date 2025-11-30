"""Pytest configuration and fixtures."""
import tempfile
import uuid
from datetime import datetime
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def temp_kv_dir(monkeypatch):
    """Create temporary directory for KV cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr("src.memory.kv_block_manager.block.KV_DATA_DIR", tmpdir)
        yield tmpdir


@pytest.fixture
def mock_openai_config():
    """Mock OpenAI configuration."""
    return {
        "api_key": "test-key",
        "base_url": "https://test.api.com",
        "model": "gpt-4o-mini"
    }


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    mock_client = Mock()
    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = "Test response"
    mock_message.tool_calls = None
    mock_response.choices = [Mock(message=mock_message)]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_model():
    """Mock transformer model."""
    mock = Mock()
    mock.device = torch.device("cpu")
    
    # Mock forward pass
    mock_output = Mock()
    mock_output.logits = torch.randn(1, 10, 1000)
    mock_output.past_key_values = [(torch.randn(1, 8, 10, 64), torch.randn(1, 8, 10, 64)) for _ in range(4)]
    mock.return_value = mock_output
    
    return mock


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer."""
    mock = Mock()
    mock.eos_token_id = 0
    mock.encode.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock.decode.return_value = "Test decoded text"
    mock.apply_chat_template.return_value = "<|im_start|>system\nTest<|im_end|>\n"
    return mock


@pytest.fixture
def sample_cache_state():
    """Sample cache state for testing."""
    return {
        "global_offset": 100,
        "saved_chunks": [
            {
                "cache": [(torch.randn(1, 8, 10, 64), torch.randn(1, 8, 10, 64)) for _ in range(4)],
                "start": 0,
                "length": 50
            },
            {
                "cache": [(torch.randn(1, 8, 10, 64), torch.randn(1, 8, 10, 64)) for _ in range(4)],
                "start": 50,
                "length": 50
            }
        ],
        "chunk_number": 2,
        "model_id": "test-model"
    }


@pytest.fixture
def block_id():
    """Generate test block ID."""
    return uuid.uuid4()


@pytest.fixture
def timestamp():
    """Generate test timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
