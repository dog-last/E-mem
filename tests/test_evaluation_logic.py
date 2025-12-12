"""Unit tests for evaluation logic."""
import os
from pathlib import Path

import pytest

os.environ.setdefault('HF_HOME', '/mnt/d/AI')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from evaluation.locomo.load_dataset import QA, load_locomo_dataset
from evaluation.locomo.utils import calculate_metrics


class TestReferenceAnswerSelection:
    def test_normal_category(self):
        qa = QA(
            question="What is the answer?",
            evidence=["D1:1"],
            category=1,
            answer="Normal answer",
            adversarial_answer=None
        )
        
        reference = qa.adversarial_answer or qa.answer if qa.category == 5 else qa.answer
        assert reference == "Normal answer"
    
    def test_adversarial_category(self):
        qa = QA(
            question="What is the answer?",
            evidence=["D1:1"],
            category=5,
            answer=None,
            adversarial_answer="Not mentioned in the conversation"
        )
        
        reference = qa.adversarial_answer or qa.answer if qa.category == 5 else qa.answer
        assert reference == "Not mentioned in the conversation"


class TestIntAnswerHandling:
    def test_int_answer(self):
        qa = QA(
            question="What year?",
            evidence=["D1:1"],
            category=2,
            answer=2022,
            adversarial_answer=None
        )
        
        metrics = calculate_metrics("The year is 2022", qa.answer)
        assert isinstance(metrics, dict)
        assert "f1" in metrics


class TestMetricsCalculation:
    @pytest.mark.parametrize("pred,ref,should_match", [
        ("2022", 2022, True),
        ("hello world", "hello world", True),
        ("Hello World", "hello world", True),
        ("the answer", "answer", True),
        ("completely different", "answer", False),
    ])
    def test_various_inputs(self, pred, ref, should_match):
        metrics = calculate_metrics(pred, ref)
        assert isinstance(metrics, dict)
        assert "exact_match" in metrics
        assert "f1" in metrics
        
        if should_match:
            assert metrics["exact_match"] == 1 or metrics["f1"] > 0.5


class TestCategory5Handling:
    @pytest.fixture
    def dataset_path(self):
        path = Path(__file__).parent.parent / "evaluation" / "eval_data" / "locomo10_part1.json"
        if not path.exists():
            pytest.skip(f"Dataset not found: {path}")
        return str(path)
    
    def test_category_5_has_reference(self, dataset_path):
        samples = load_locomo_dataset(dataset_path)
        
        cat5_found = False
        for sample in samples:
            for qa in sample.qa:
                if qa.category == 5:
                    cat5_found = True
                    reference = qa.adversarial_answer or qa.answer
                    assert reference is not None
                    
                    metrics = calculate_metrics("Not mentioned", reference)
                    assert isinstance(metrics, dict)
                    break
            if cat5_found:
                break
        
        assert cat5_found
