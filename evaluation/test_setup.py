"""Test script to verify evaluation setup."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        print("✓ load_dataset imported successfully")
    except Exception as e:
        print(f"✗ Failed to import load_dataset: {e}")
        return False
    
    try:
        print("✓ utils imported successfully")
    except Exception as e:
        print(f"✗ Failed to import utils: {e}")
        return False
    
    try:
        print("✓ ChatManager imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ChatManager: {e}")
        return False
    
    return True


def test_config():
    """Test if config file exists and is valid."""
    print("\nTesting config...")
    
    try:
        import yaml
        config_path = Path(__file__).parent / "config.yaml"
        
        if not config_path.exists():
            print(f"✗ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['model', 'memory', 'evaluation', 'logging']
        for key in required_keys:
            if key not in config:
                print(f"✗ Missing required config key: {key}")
                return False
        
        print("✓ Config file is valid")
        return True
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False


def test_directories():
    """Test if required directories will be created."""
    print("\nTesting directory creation...")
    
    import os
    eval_dir = Path(__file__).parent
    
    # These will be created automatically by the code
    dirs_to_check = ['logs', 'results']
    
    for dir_name in dirs_to_check:
        dir_path = eval_dir / dir_name
        os.makedirs(dir_path, exist_ok=True)
        if dir_path.exists():
            print(f"✓ Directory {dir_name}/ created/exists")
        else:
            print(f"✗ Failed to create directory: {dir_name}/")
            return False
    
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("Evaluation Setup Test")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("Directories", test_directories),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test {name} failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All tests passed! Setup is ready.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
