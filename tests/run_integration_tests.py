# tests/run_integration_tests.py

import asyncio
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def run_integration_tests():
    """Run complete integration test suite"""
    print("=== SOLITON MEMORY INTEGRATION TESTS ===\n")
    
    # Test modules
    test_modules = [
        "tests.test_dark_solitons",
        "tests.test_topology_morphing", 
        "tests.test_memory_consolidation",
    ]
    
    # Run each test module
    for module in test_modules:
        print(f"\nRunning {module}...")
        result = pytest.main(["-v", f"{module.replace('.', '/')}.py"])
        
        if result != 0:
            print(f"❌ {module} failed")
            return False
            
    print("\n✅ All integration tests passed!")
    
    # Run benchmarks
    print("\n=== RUNNING BENCHMARKS ===")
    from benchmarks.benchmark_soliton_performance import main as run_benchmarks
    await run_benchmarks()
    
    return True

if __name__ == "__main__":
    success = asyncio.run(run_integration_tests())
    sys.exit(0 if success else 1)
