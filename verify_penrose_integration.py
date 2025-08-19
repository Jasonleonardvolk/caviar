"""
Quick verification script for Penrose engine integration
"""
import sys
import time
import numpy as np
from pathlib import Path

print("🔍 Penrose Engine Verification")
print("=" * 50)

# Check if Penrose is available
try:
    from concept_mesh.similarity import penrose_available, penrose, compute_similarity
    print(f"✅ concept_mesh.similarity imported successfully")
    print(f"   Penrose available: {penrose_available}")
    
    if penrose:
        info = penrose.get_info() if hasattr(penrose, 'get_info') else "No info available"
        print(f"   Engine info: {info}")
except ImportError as e:
    print(f"❌ Failed to import concept_mesh.similarity: {e}")
    sys.exit(1)

# Try direct import
print("\n📦 Direct import test:")
try:
    import penrose_engine_rs
    print("✅ penrose_engine_rs module found")
    print(f"   Module path: {penrose_engine_rs.__file__}")
except ImportError:
    print("❌ penrose_engine_rs not found (using Python fallback)")

# Performance test
print("\n⚡ Performance test:")
if penrose_available:
    try:
        # Create test data
        embeddings = np.random.rand(1000, 256).astype(np.float64)
        queries = np.random.rand(10, 256).astype(np.float64)
        
        # Time the operation
        start = time.perf_counter()
        similarities = compute_similarity(embeddings, queries, threshold=0.7)
        elapsed = time.perf_counter() - start
        
        print(f"✅ Computed similarities for 10 queries against 1000 embeddings")
        print(f"   Shape: {similarities.shape}")
        print(f"   Time: {elapsed*1000:.2f}ms")
        print(f"   Speed: {(10*1000)/(elapsed*1000):.0f} comparisons/ms")
        
        # Check if it's actually fast (Rust should be < 10ms for this size)
        if elapsed < 0.01:
            print(f"   🚀 This is definitely the Rust engine!")
        elif elapsed < 0.1:
            print(f"   ⚡ Good performance, likely Rust")
        else:
            print(f"   🐢 Slow performance, might be Python fallback")
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
else:
    print("⚠️  Using Python fallback, expect slower performance")

# Check logs location
print("\n📋 Next steps:")
print("1. Run: python enhanced_launcher.py")
print("2. Check the logs for:")
print("   - 'Penrose engine initialized (Rust, SIMD, rank=32)' ✅")
print("   - No '⚠️ Penrose not available' warning")
print("3. Watch for non-zero oscillator counts in lattice runner")

print("\n✨ Verification complete!")