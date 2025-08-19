"""
Quick test to ensure Rust code compiles and basic functionality works
Run this after building with maturin
"""

def test_penrose_engine():
    print("Testing Penrose Engine...")
    
    try:
        import penrose_engine_rs as penrose
        print("✅ Module imported successfully")
        
        # Test basic info
        info = penrose.get_info()
        print(f"✅ Engine info: {info}")
        
        # Test availability
        available = penrose.is_available()
        print(f"✅ Engine available: {available}")
        
        # Test projection
        import numpy as np
        test_matrix = np.random.rand(10, 20)
        projected = penrose.project(test_matrix, 5)
        print(f"✅ Projection test: {test_matrix.shape} -> {projected.shape}")
        
        # Test batch similarity
        embeddings = np.random.rand(50, 32)
        queries = np.random.rand(5, 32)
        similarities = penrose.batch_similarity(embeddings, queries, 0.7)
        print(f"✅ Similarity test: {similarities.shape}")
        
        # Test sparse projection
        indices = [(0, 0), (1, 1), (2, 3), (4, 5)]
        values = [1.0, 2.0, 3.0, 4.0]
        shape = (10, 10)
        sparse_indices, sparse_values = penrose.project_sparse(indices, values, shape, 3)
        print(f"✅ Sparse projection: {len(indices)} -> {len(sparse_indices)} entries")
        
        print("\n🎉 All tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Make sure you've run: maturin develop --release")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_penrose_engine()
    exit(0 if success else 1)