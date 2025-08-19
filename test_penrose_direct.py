"""Test Penrose performance without pytest."""
import time
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from concept_mesh.similarity import penrose
    
    print("Testing Penrose performance...")
    print("=" * 50)
    
    # Test 1: Single similarity
    a = np.random.rand(512).astype("float32")
    b = np.random.rand(512).astype("float32")
    
    t0 = time.perf_counter()
    for _ in range(10_000):
        penrose.compute_similarity(a.tolist(), b.tolist())
    elapsed = time.perf_counter() - t0
    
    print(f"✅ Single similarity test: {elapsed:.3f}s for 10k operations")
    print(f"   Average: {elapsed/10_000*1000:.3f}ms per operation")
    
    if elapsed < 0.3:
        print("   PASSED: Performance is good!")
    else:
        print("   WARNING: Performance may be degraded")
    
    # Test 2: Batch similarity
    print("\nTesting batch similarity...")
    query = np.random.rand(512).astype("float32")
    corpus = [np.random.rand(512).astype("float32").tolist() for _ in range(1000)]
    
    t0 = time.perf_counter()
    results = penrose.batch_similarity(query.tolist(), corpus)
    elapsed = time.perf_counter() - t0
    
    print(f"✅ Batch similarity test: {elapsed:.3f}s for 1k vectors")
    
    if elapsed < 0.05:
        print("   PASSED: Batch performance is good!")
    else:
        print("   WARNING: Batch performance may be degraded")
    
    print("\n" + "=" * 50)
    print("All Penrose tests completed!")
    
except ImportError as e:
    print(f"❌ Error: Could not import Penrose: {e}")
    print("Make sure you're in the activated venv and Penrose is built")
    sys.exit(1)
