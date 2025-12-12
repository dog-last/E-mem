"""Unit tests for evaluation dataset loading."""
from pathlib import Path

import pytest

from evaluation.locomo.load_dataset import QA, Sample, load_locomo_dataset


@pytest.fixture
def dataset_path():
    path = Path(__file__).parent.parent / "evaluation" / "locomo" / "eval_data" / "locomo10_part1.json"
    if not path.exists():
        pytest.skip(f"Dataset not found: {path}")
    return str(path)


class TestLoadDataset:
    def test_load_basic(self, dataset_path):
        samples = load_locomo_dataset(dataset_path)
        assert len(samples) > 0
        assert isinstance(samples[0], Sample)
    
    def test_qa_structure(self, dataset_path):
        samples = load_locomo_dataset(dataset_path)
        
        total_qa = 0
        categories = set()
        
        for sample in samples:
            assert hasattr(sample, 'qa')
            assert hasattr(sample, 'conversation')
            
            for qa in sample.qa:
                total_qa += 1
                assert isinstance(qa, QA)
                assert hasattr(qa, 'question')
                assert hasattr(qa, 'category')
                assert hasattr(qa, 'evidence')
                categories.add(qa.category)
        
        assert total_qa > 0
        assert categories.issubset({1, 2, 3, 4, 5})
    
    def test_conversation_structure(self, dataset_path):
        samples = load_locomo_dataset(dataset_path)
        sample = samples[0]
        
        conv = sample.conversation
        assert hasattr(conv, 'speaker_a')
        assert hasattr(conv, 'speaker_b')
        assert hasattr(conv, 'sessions')
        assert len(conv.sessions) > 0
        
        for session_key, session in conv.sessions.items():
            assert hasattr(session, 'date_time')
            assert hasattr(session, 'turns')
            assert len(session.turns) > 0
            
            for turn in session.turns:
                assert hasattr(turn, 'speaker')
                assert hasattr(turn, 'text')
                assert hasattr(turn, 'dia_id')
    
    def test_category_5_has_adversarial(self, dataset_path):
        samples = load_locomo_dataset(dataset_path)
        
        cat5_count = 0
        cat5_with_adversarial = 0
        
        for sample in samples:
            for qa in sample.qa:
                if qa.category == 5:
                    cat5_count += 1
                    if qa.adversarial_answer:
                        cat5_with_adversarial += 1
        
        if cat5_count > 0:
            assert cat5_with_adversarial > 0
