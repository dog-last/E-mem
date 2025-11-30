"""Test script to verify dataset loading."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from evaluation.load_dataset import load_locomo_dataset


def test_load():
    dataset_path = Path(__file__).parent / "eval_data" / "locomo10_part1.json"
    print(f"Loading dataset from: {dataset_path}")
    
    try:
        samples = load_locomo_dataset(str(dataset_path))
        print(f"✓ Successfully loaded {len(samples)} samples")
        
        # Check QA items
        total_qa = 0
        qa_with_answer = 0
        qa_with_adversarial = 0
        category_counts = {}
        
        for sample in samples:
            for qa in sample.qa:
                total_qa += 1
                if qa.answer:
                    qa_with_answer += 1
                if qa.adversarial_answer:
                    qa_with_adversarial += 1
                
                cat = qa.category
                category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print("\nQA Statistics:")
        print(f"  Total QA items: {total_qa}")
        print(f"  With answer: {qa_with_answer}")
        print(f"  With adversarial_answer: {qa_with_adversarial}")
        
        print("\nCategory Distribution:")
        for cat in sorted(category_counts.keys()):
            print(f"  Category {cat}: {category_counts[cat]}")
        
        # Show example of each type
        print("\n--- Example QA Items ---")
        for sample in samples[:1]:
            for qa in sample.qa[:5]:
                print(f"\nCategory {qa.category}:")
                print(f"  Question: {qa.question}")
                print(f"  Answer: {qa.answer}")
                print(f"  Adversarial: {qa.adversarial_answer}")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_load()
    sys.exit(0 if success else 1)
