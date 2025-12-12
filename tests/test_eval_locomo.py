"""Unit tests for eval_locomo.py."""
import logging
import os
from unittest.mock import Mock, patch

import pytest

os.environ.setdefault('HF_HOME', '/mnt/d/AI')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')


@pytest.fixture
def mock_config():
    return {
        'model': {
            'model_id': 'test-model',
            'openai_config': None,
            'model_context_window': 4096,
            'attn_implementation': 'sdpa',
            'device_map': 'auto',
            'quantization_config': None
        },
        'memory': {
            'clean_cache_first': True,
            'router_system_prompt': None
        },
        'locomo_eval': {
            'dataset_path': 'eval_data/test.json',
            'output_dir': 'results',
            'ratio': 1.0,
            'conversation_auto_save': False,
            'categories': [1, 2, 3, 4, 5]
        },
        'logging': {
            'log_dir': 'logs',
            'log_level': 'INFO'
        }
    }


@pytest.fixture
def mock_logger():
    return Mock()


@pytest.fixture
def mock_samples():
    from evaluation.locomo.load_dataset import QA, Conversation, Sample, Session, Turn
    
    turn = Turn(speaker="A", dia_id="D1:1", text="Test conversation")
    session = Session(date_time="2023-01-01", turns=[turn])
    conversation = Conversation(
        speaker_a="Alice",
        speaker_b="Bob",
        sessions={"session_1": session}
    )
    
    qa1 = QA(question="Q1?", evidence=["D1:1"], category=1, answer="A1")
    qa2 = QA(question="Q2?", evidence=["D1:2"], category=2, answer=2022)
    qa3 = QA(question="Q3?", evidence=["D1:3"], category=5, adversarial_answer="Not mentioned")
    
    sample = Sample(qa=[qa1, qa2, qa3], conversation=conversation)
    return [sample]


class TestSetupLogger:
    def test_setup_logger_basic(self, tmp_path):
        from evaluation.locomo.eval_locomo import setup_logger
        
        log_file = str(tmp_path / "test.log")
        logger = setup_logger(log_file)
        
        assert logger is not None
        assert logger.name == 'locomo_eval'
        
        # Check root logger has handlers (console + file)
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) >= 2


class TestEvaluateDataset:
    @patch('evaluation.locomo.eval_locomo.load_locomo_dataset')
    @patch('evaluation.locomo.eval_locomo.create_chat_manager')
    @patch('evaluation.locomo.eval_locomo.calculate_metrics')
    def test_evaluate_dataset_basic(self, mock_calc_metrics, mock_create_chat_manager, 
                                    mock_load_dataset, mock_config, mock_logger, 
                                    mock_samples, tmp_path):
        from evaluation.locomo.eval_locomo import evaluate_dataset

        # Setup mocks
        mock_load_dataset.return_value = mock_samples
        mock_agent = Mock()
        mock_agent.chat.return_value = "Test answer"
        mock_agent.last_queried_memory = "Test memory content"  # Add this
        mock_create_chat_manager.return_value = mock_agent
        mock_calc_metrics.return_value = {
            'exact_match': 1, 'f1': 0.9, 'rouge1_f': 0.8,
            'rouge2_f': 0.7, 'rougeL_f': 0.75, 'bleu1': 0.85,
            'bleu2': 0.8, 'bleu3': 0.75, 'bleu4': 0.7,
            'meteor': 0.8, 'sbert_similarity': 0.9
        }
        
        # Update config paths
        mock_config['locomo_eval']['output_dir'] = str(tmp_path / 'results')
        mock_config['logging']['log_dir'] = str(tmp_path / 'logs')
        
        # Run evaluation
        results = evaluate_dataset(mock_config, mock_logger)
        
        # Assertions
        assert results is not None
        assert 'total_questions' in results
        assert results['total_questions'] == 3
        assert 'aggregate_metrics' in results
        assert 'individual_results' in results
        assert len(results['individual_results']) == 3
    
    @patch('evaluation.locomo.eval_locomo.load_locomo_dataset')
    @patch('evaluation.locomo.eval_locomo.create_chat_manager')
    def test_evaluate_dataset_with_ratio(self, mock_create_chat_manager, mock_load_dataset,
                                         mock_config, mock_logger, mock_samples):
        from evaluation.locomo.eval_locomo import evaluate_dataset

        # Create multiple samples
        mock_load_dataset.return_value = mock_samples * 10
        mock_config['locomo_eval']['ratio'] = 0.1
        
        mock_agent = Mock()
        mock_agent.chat.return_value = "Test"
        mock_agent.last_queried_memory = "Test memory"
        mock_create_chat_manager.return_value = mock_agent
        
        with patch('evaluation.locomo.eval_locomo.calculate_metrics') as mock_calc:
            mock_calc.return_value = {'f1': 0.5, 'exact_match': 0}
            with patch('builtins.open', create=True):
                evaluate_dataset(mock_config, mock_logger)
        
        # Should only process 1 sample (10 * 0.1 = 1)
        assert mock_create_chat_manager.call_count == 1
    
    @patch('evaluation.locomo.eval_locomo.load_locomo_dataset')
    @patch('evaluation.locomo.eval_locomo.create_chat_manager')
    @patch('evaluation.locomo.eval_locomo.calculate_metrics')
    def test_category_5_handling(self, mock_calc_metrics, mock_create_chat_manager,
                                 mock_load_dataset, mock_config, mock_logger, 
                                 mock_samples):
        from evaluation.locomo.eval_locomo import evaluate_dataset
        
        mock_load_dataset.return_value = mock_samples
        mock_agent = Mock()
        mock_agent.chat.return_value = "Not mentioned"
        mock_agent.last_queried_memory = "Test memory"
        mock_create_chat_manager.return_value = mock_agent
        mock_calc_metrics.return_value = {'f1': 1.0, 'exact_match': 1}
        
        with patch('builtins.open', create=True):
            results = evaluate_dataset(mock_config, mock_logger)
        
        # Check that category 5 question was processed
        cat5_results = [r for r in results['individual_results'] if r['category'] == 5]
        assert len(cat5_results) == 1
        assert "Not mentioned" in cat5_results[0]['reference']
    
    @patch('evaluation.locomo.eval_locomo.load_locomo_dataset')
    @patch('evaluation.locomo.eval_locomo.create_chat_manager')
    def test_conversation_auto_save(self, mock_create_chat_manager, mock_load_dataset,
                                    mock_config, mock_logger, mock_samples):
        from evaluation.locomo.eval_locomo import evaluate_dataset
        
        mock_load_dataset.return_value = mock_samples
        mock_agent = Mock()
        mock_agent.chat.return_value = "Answer"
        mock_agent.last_queried_memory = "Test memory"
        mock_create_chat_manager.return_value = mock_agent
        
        # Enable auto_save for conversations
        mock_config['locomo_eval']['conversation_auto_save'] = True
        
        with patch('evaluation.locomo.eval_locomo.calculate_metrics') as mock_calc:
            mock_calc.return_value = {'f1': 0.5}
            with patch('builtins.open', create=True):
                evaluate_dataset(mock_config, mock_logger)
        
        # Check that chat was called with auto_save=True for conversations
        conversation_calls = [call for call in mock_agent.chat.call_args_list 
                            if call[1].get('auto_save')]
        assert len(conversation_calls) > 0


class TestMain:
    @patch('evaluation.locomo.eval_locomo.evaluate_dataset')
    @patch('evaluation.locomo.eval_locomo.setup_logger')
    @patch('builtins.open')
    @patch('evaluation.locomo.eval_locomo.yaml.safe_load')
    def test_main_basic(self, mock_yaml_load, mock_open, mock_setup_logger,
                       mock_evaluate, mock_config):
        import sys

        from evaluation.locomo.eval_locomo import main
        
        mock_yaml_load.return_value = mock_config
        mock_logger = Mock()
        mock_setup_logger.return_value = mock_logger
        mock_evaluate.return_value = {'total_questions': 10}
        
        # Mock command line args
        test_args = ['eval_locomo.py', '--config', 'config.yaml']
        with patch.object(sys, 'argv', test_args):
            with patch('os.makedirs'):
                main()
        
        mock_evaluate.assert_called_once()
