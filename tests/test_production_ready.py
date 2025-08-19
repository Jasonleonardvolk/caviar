# tests/test_production_ready.py - Basic production tests
import pytest
import asyncio
import numpy as np
from core.embedding_client import embed_concepts
from core.penrose_verifier_enhanced import get_penrose_verifier

@pytest.mark.asyncio
async def test_local_embedding():
    """Test local embedding service"""
    test_concepts = ["Koopman operator", "spectral analysis"]
    
    embeddings = await embed_concepts(test_concepts)
    
    # Check embeddings are valid
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0  # Should have dimensions
    
    # Check vectors are normalized
    for vec in embeddings:
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.1, f"Vector not normalized: {norm}"
    
    # Check similarity is reasonable
    similarity = np.dot(embeddings[0], embeddings[1])
    assert similarity > 0.65, f"Concepts should be related: {similarity}"

def test_penrose_verification():
    """Test Penrose verification with quality gates"""
    verifier = get_penrose_verifier()
    
    # Create test embeddings
    embeddings = np.random.randn(10, 1536)
    # Normalize them
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    concepts = [f"concept_{i}" for i in range(10)]
    
    result = verifier.verify_tessera(embeddings, concepts)
    
    # Check result structure
    assert result.status in ["VERIFIED", "WARNING", "FAILED"]
    assert 0 <= result.geometric_score <= 1
    assert 0 <= result.phase_coherence <= 1
    assert 0 <= result.semantic_stability <= 1
    assert isinstance(result.vector_quality, dict)
    assert len(result.tessera_digest) == 64  # SHA-256 hex

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
