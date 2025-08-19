# tests/test_smoke_prod.py - Production smoke test
import httpx
import numpy as np

def test_embed_and_penrose():
    """Complete production smoke test"""
    
    # Test embedding endpoint
    jwt = {"Authorization": "Bearer dev-token"}  # skip if DISABLE_AUTH
    sample = ["Koopman operator synthesises spectral modes."]
    
    with httpx.Client() as client:
        r = client.post(
            "http://localhost:8080/embed",
            json={"texts": sample},
            headers=jwt
        )
        
        assert r.status_code == 200, f"Embedding failed: {r.text}"
        
        vec = np.array(r.json()["embeddings"][0])
        assert abs(np.linalg.norm(vec) - 1) < 1e-2, f"Vector not normalized: norm={np.linalg.norm(vec)}"
    
    # Quick Penrose check (no mesh)
    from core.penrose_verifier_production import get_penrose_verifier
    res = get_penrose_verifier().verify_tessera(vec.reshape(1, -1), sample)
    assert res.status == "VERIFIED", f"Penrose failed: {res.status}"
    
    print("✅ All smoke tests passed!")

if __name__ == "__main__":
    test_embed_and_penrose()
