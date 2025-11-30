"""
Installation test script.
Run this to verify that all components are properly installed and configured.
"""

import os
import sys


def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        
        import transformers
        print(f"  ✓ Transformers {transformers.__version__}")
        
        from openai import OpenAI
        _ = OpenAI
        print("  ✓ OpenAI library")
        
        from src.memory.kv_block_manager.block import KVBlock, clear_cache
        _ = (KVBlock, clear_cache)
        print("  ✓ KVBlock")
        
        from src.memory.memory_agent.agent import MemoryAgent
        _ = MemoryAgent
        print("  ✓ MemoryAgent")
        
        from src.memory.router.router import Router
        _ = Router
        print("  ✓ Router")
        
        from src.memory.core.loop_handler import MemoryHandler
        _ = MemoryHandler
        print("  ✓ MemoryHandler")
        
        from src.conversation_manager.chat_handler import ChatManager
        _ = ChatManager
        print("  ✓ ChatManager")
        
        from src.agent.base import BaseAgent
        _ = BaseAgent
        print("  ✓ BaseAgent")
        
        print("\n✅ All imports successful!\n")
        return True
    except ImportError as e:
        print(f"\n❌ Import failed: {e}\n")
        return False


def test_kv_data_directory():
    """Test if kv_data directory is created."""
    print("Testing kv_data directory creation...")
    from src.memory.kv_block_manager.block import KV_DATA_DIR
    
    if os.path.exists(KV_DATA_DIR):
        print(f"  ✓ Directory exists: {KV_DATA_DIR}")
        print("\n✅ Directory test passed!\n")
        return True
    else:
        print(f"  ❌ Directory not found: {KV_DATA_DIR}")
        print("\n❌ Directory test failed!\n")
        return False


def test_config():
    """Test if config file exists."""
    print("Testing configuration...")
    try:
        import config
        print("  ✓ config.py found")
        print(f"  ✓ MODEL_ID: {config.MODEL_ID}")
        print(f"  ✓ OPENAI_CONFIG: {'api_key' in config.OPENAI_CONFIG}")
        print("\n✅ Configuration test passed!\n")
        return True
    except ImportError:
        print("  ⚠️  config.py not found (using defaults)")
        print("  ℹ️  Copy config.example.py to config.py to customize")
        print("\n⚠️  Configuration test skipped!\n")
        return True  # Not critical


def test_cuda():
    """Test CUDA availability."""
    print("Testing CUDA...")
    import torch
    
    if torch.cuda.is_available():
        print("  ✓ CUDA available")
        print(f"  ✓ Device count: {torch.cuda.device_count()}")
        print(f"  ✓ Current device: {torch.cuda.current_device()}")
        print(f"  ✓ Device name: {torch.cuda.get_device_name(0)}")
        print("\n✅ CUDA test passed!\n")
    else:
        print("  ⚠️  CUDA not available (will use CPU)")
        print("  ℹ️  Performance will be slower on CPU")
        print("\n⚠️  CUDA test skipped!\n")
    
    return True


def test_basic_functionality():
    """Test basic KVBlock functionality."""
    print("Testing basic functionality...")
    try:
        import uuid
        from datetime import datetime

        from src.memory.kv_block_manager.block import KVBlock

        # Create a test block
        block = KVBlock(
            block_id=uuid.uuid4(),
            create_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            block_size=1000
        )
        print("  ✓ KVBlock created")
        
        # Test save
        test_cache = {
            "global_offset": 100,
            "saved_chunks": [],
            "chunk_number": 1,
            "model_id": "test"
        }
        is_full = block.save_cache(test_cache, 50)
        print(f"  ✓ Cache saved (full: {is_full})")
        
        # Test load
        loaded = block.load_cache()
        assert loaded["global_offset"] == 100
        print("  ✓ Cache loaded")
        
        # Clean up
        import os
        if os.path.exists(block.store_target):
            os.remove(block.store_target)
        print("  ✓ Test file cleaned up")
        
        print("\n✅ Basic functionality test passed!\n")
        return True
    except Exception as e:
        print(f"\n❌ Basic functionality test failed: {e}\n")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("KV-Cached Memory Agent - Installation Test")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("KV Data Directory", test_kv_data_directory()))
    results.append(("Configuration", test_config()))
    results.append(("CUDA", test_cuda()))
    results.append(("Basic Functionality", test_basic_functionality()))
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    print()
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("🎉 All tests passed! System is ready to use.")
        print()
        print("Next steps:")
        print("  1. Copy config.example.py to config.py")
        print("  2. Edit config.py with your settings")
        print("  3. Run: python main.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print()
        print("Common fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Check CUDA installation if GPU tests failed")
        print("  - Verify file permissions for kv_data directory")
    
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
