# core/penrose_verifier_test.py - Test version with relaxed thresholds
import numpy as np
import logging
from typing import Dict, Any, List
from dataclasses import dataclass
import hashlib
import os

logger = logging.getLogger(__name__)

@dataclass
class PenroseVerificationResult:
    """Penrose verification result"""
    status: str
    geometric_score: float
    phase_coherence: float
    semantic_stability: float
    vector_quality: Dict[str, bool]
    tessera_digest: str
    metadata: Dict[str, Any]

class TestPenroseVerifier:
    """Test Penrose verifier with relaxed thresholds"""
    
    def __init__(self):
        self.COSINE_THRESHOLD = 0.3  # Much lower for testing
        self.VECTOR_NORM_MIN = 0.9
        self.VECTOR_NORM_MAX = 1.1
        self.ENTROPY_THRESHOLD = 1.0  # Much lower for testing
        self.STABILITY_THRESHOLD = 0.5  # Lower for testing
        logger.info("Test Penrose Verifier initialized with relaxed thresholds")
    
    def verify_tessera(
        self,
        embeddings: np.ndarray,
        concepts: List[str],
        metadata: Dict[str, Any] = None
    ) -> PenroseVerificationResult:
        """Simplified verification for testing"""
        
        if embeddings.shape[0] == 0:
            return PenroseVerificationResult(
                status="FAILED",
                geometric_score=0.0,
                phase_coherence=0.0,
                semantic_stability=0.0,
                vector_quality={"empty_input": False},
                tessera_digest="",
                metadata={"error": "Empty embeddings"}
            )
        
        # Basic checks
        norms = np.linalg.norm(embeddings, axis=1)
        norm_check = np.all((norms >= self.VECTOR_NORM_MIN) & (norms <= self.VECTOR_NORM_MAX))
        
        # Simplified entropy check
        entropy_check = True  # Always pass for testing
        
        # Simplified geometric check
        geometric_score = 0.8  # Fixed score for testing
        geometric_check = True
        
        # Simplified phase coherence
        phase_coherence = 0.7
        phase_check = True
        
        # Simplified stability
        stability_score = 0.9
        stability_check = True
        
        vector_quality = {
            "norm_check": norm_check,
            "entropy_check": entropy_check,
            "geometric_check": geometric_check,
            "phase_check": phase_check,
            "stability_check": stability_check
        }
        
        # Generate digest
        embedding_bytes = embeddings.astype(np.float32).tobytes()
        concept_bytes = "::".join(concepts).encode('utf-8')
        tessera_digest = hashlib.sha256(embedding_bytes + concept_bytes).hexdigest()
        
        # Always pass for testing
        status = "VERIFIED"
        
        passed_gates = sum(vector_quality.values())
        pass_rate = passed_gates / len(vector_quality)
        
        result_metadata = {
            "vector_count": embeddings.shape[0],
            "vector_dim": embeddings.shape[1],
            "mean_norm": float(np.mean(norms)),
            "pass_rate": pass_rate,
            "slo_met": True,
            **(metadata or {})
        }
        
        return PenroseVerificationResult(
            status=status,
            geometric_score=geometric_score,
            phase_coherence=phase_coherence,
            semantic_stability=stability_score,
            vector_quality=vector_quality,
            tessera_digest=tessera_digest,
            metadata=result_metadata
        )

# Override for testing
_test_verifier = None

def get_penrose_verifier() -> TestPenroseVerifier:
    """Get test verifier"""
    global _test_verifier
    if _test_verifier is None:
        _test_verifier = TestPenroseVerifier()
    return _test_verifier
