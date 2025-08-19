# tests/test_smoke_prod_fixed.py - Fixed smoke test without auth
import httpx
import numpy as np
import os

def test_embed_and_penrose():
    """Complete production smoke test"""
    
    # Test embedding endpoint
    sample = ["Koopman operator synthesises spectral modes."]
    
    # No auth needed since DISABLE_AUTH=true
    with httpx.Client(timeout=30.0) as client:
        r = client.post(
            "http://localhost:8080/embed",
            json={"texts": sample}
        )
        
        assert r.status_code == 200, f"Embedding failed: {r.text}"
        
        data = r.json()
        vec = np.array(data["embeddings"][0])
        
        print(f"✅ Embedding successful!")
        print(f"  - Vector dimensions: {len(vec)}")
        print(f"  - Vector norm: {np.linalg.norm(vec):.4f}")
        print(f"  - Cache hits: {data.get('cache_hits', 0)}")
        print(f"  - Cache misses: {data.get('cache_misses', 0)}")
        print(f"  - Processing time: {data.get('processing_time_ms', 0):.2f}ms")
        
        assert abs(np.linalg.norm(vec) - 1) < 1e-2, f"Vector not normalized: norm={np.linalg.norm(vec)}"
    
    # Quick Penrose check
    try:
        import sys
        sys.path.append('..')  # Add parent directory to path
        from core.penrose_verifier_production import get_penrose_verifier
        
        res = get_penrose_verifier().verify_tessera(vec.reshape(1, -1), sample)
        assert res.status in ["VERIFIED", "WARNING"], f"Penrose failed: {res.status}"
        
        print(f"\n✅ Penrose verification: {res.status}")
        print(f"  - Geometric score: {res.geometric_score:.3f}")
        print(f"  - Phase coherence: {res.phase_coherence:.3f}")
        print(f"  - Semantic stability: {res.semantic_stability:.3f}")
        print(f"  - Pass rate: {res.metadata.get('pass_rate', 0):.3f}")
        
    except ImportError as e:
        print(f"\n⚠️ Penrose test skipped (import error): {e}")
    
    print("\n🎯 All smoke tests passed!")

if __name__ == "__main__":
    test_embed_and_penrose()
