"""Direct test for Penrose performance without pytest infrastructure"""
import time
import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from concept_mesh.similarity import penrose
    print("✅ Successfully imported Penrose")
    
    # Test 1: Single similarity performance
    print("\nTest 1: Single similarity performance")
    print("-" * 40)
    a = np.random.rand(512).astype("float32")
    b = np.random.rand(512).astype("float32")
    
    # Warmup
    for _ in range(100):
        penrose.compute_similarity(a, b)
    
    # Actual test
    t0 = time.perf_counter()
    iterations = 10_000
    for _ in range(iterations):
        penrose.compute_similarity(a, b)
    elapsed = time.perf_counter() - t0
    
    print(f"Time for {iterations} operations: {elapsed:.3f}s")
    print(f"Average per operation: {(elapsed/iterations)*1000:.3f}ms")
    
    if elapsed < 0.3:
        print("✅ PASSED: Performance is within threshold (<0.3s)")
    else:
        print("❌ FAILED: Performance exceeded threshold (>0.3s)")
    
    # Test 2: Batch similarity performance  
    print("\nTest 2: Batch similarity performance")
    print("-" * 40)
    query = np.random.rand(512).astype("float32")
    corpus = [np.random.rand(512).astype("float32") for _ in range(1000)]
    
    t0 = time.perf_counter()
    results = penrose.batch_similarity(query, corpus)
    elapsed = time.perf_counter() - t0
    
    print(f"Time for batch (1000 vectors): {elapsed:.3f}s")
    print(f"Results length: {len(results)}")
    
    if elapsed < 0.1:
        print("✅ PASSED: Batch performance is within threshold (<0.1s)")
    else:
        print("❌ FAILED: Batch performance exceeded threshold (>0.1s)")
        
    print("\n✅ All Penrose tests completed!")
    
except ImportError as e:
    print(f"❌ Error: Could not import Penrose: {e}")
    print("\nMake sure you have:")
    print("1. Activated the virtual environment")
    print("2. Built Penrose with: maturin develop --release")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
