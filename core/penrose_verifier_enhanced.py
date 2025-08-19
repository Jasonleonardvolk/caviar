# core/penrose_verifier_enhanced.py - Production Penrose with quality gates
import numpy as np
import logging
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass
import hashlib
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import os

logger = logging.getLogger(__name__)

@dataclass
class PenroseVerificationResult:
    """Comprehensive Penrose verification result"""
    status: str  # "VERIFIED", "FAILED", "WARNING"
    geometric_score: float
    phase_coherence: float
    semantic_stability: float
    vector_quality: Dict[str, bool]
    tessera_digest: str
    metadata: Dict[str, Any]

class EnhancedPenroseVerifier:
    """Production Penrose verifier with strict quality gates"""
    
    def __init__(self):
        # Enhanced quality thresholds
        self.COSINE_THRESHOLD = float(os.getenv("TORI_COSINE_THRESHOLD", "0.65"))
        self.VECTOR_NORM_MIN = 0.9
        self.VECTOR_NORM_MAX = 1.1
        self.ENTROPY_THRESHOLD = 4.0
        self.STABILITY_THRESHOLD = 0.92
        self.GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
        
        logger.info(f"Enhanced Penrose Verifier initialized with cosine threshold: {self.COSINE_THRESHOLD}")
    
    def verify_tessera(
        self,
        embeddings: np.ndarray,
        concepts: List[str],
        metadata: Dict[str, Any] = None
    ) -> PenroseVerificationResult:
        """Comprehensive tessera verification with quality gates"""
        
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
        
        # Quality Gate 1: Vector norm validation
        norms = np.linalg.norm(embeddings, axis=1)
        norm_check = np.all((norms >= self.VECTOR_NORM_MIN) & (norms <= self.VECTOR_NORM_MAX))
        
        # Quality Gate 2: Entropy validation (semantic richness)
        entropies = [self._calculate_shannon_entropy(emb) for emb in embeddings]
        entropy_check = np.all(np.array(entropies) > self.ENTROPY_THRESHOLD)
        
        # Quality Gate 3: Geometric verification
        geometric_score = self._verify_geometric_properties(embeddings)
        geometric_check = geometric_score > 0.7
        
        # Quality Gate 4: Phase coherence
        phase_coherence = self._compute_phase_coherence(embeddings)
        phase_check = phase_coherence > 0.6
        
        # Quality Gate 5: Semantic stability (if re-embedding is available)
        stability_score = self._estimate_semantic_stability(embeddings, concepts)
        stability_check = stability_score > self.STABILITY_THRESHOLD
        
        # Aggregate quality checks
        vector_quality = {
            "norm_check": norm_check,
            "entropy_check": entropy_check,
            "geometric_check": geometric_check,
            "phase_check": phase_check,
            "stability_check": stability_check
        }
        
        # Generate tessera digest
        tessera_digest = self._generate_tessera_digest(embeddings, concepts)
        
        # Determine overall status
        critical_failures = not (norm_check and entropy_check and geometric_check)
        if critical_failures:
            status = "FAILED"
        elif not (phase_check and stability_check):
            status = "WARNING"
        else:
            status = "VERIFIED"
        
        result_metadata = {
            "vector_count": embeddings.shape[0],
            "vector_dim": embeddings.shape[1],
            "mean_norm": float(np.mean(norms)),
            "mean_entropy": float(np.mean(entropies)),
            "cosine_threshold": self.COSINE_THRESHOLD,
            **(metadata or {})
        }
        
        verification_result = PenroseVerificationResult(
            status=status,
            geometric_score=geometric_score,
            phase_coherence=phase_coherence,
            semantic_stability=stability_score,
            vector_quality=vector_quality,
            tessera_digest=tessera_digest,
            metadata=result_metadata
        )
        
        logger.info(f"Penrose verification complete: {status} (geometric={geometric_score:.3f})")
        
        if status == "FAILED":
            failed_checks = [k for k, v in vector_quality.items() if not v]
            raise ValueError(f"Penrose verification FAILED. Failed checks: {failed_checks}")
        
        return verification_result
    
    def _calculate_shannon_entropy(self, vector: np.ndarray) -> float:
        """Calculate Shannon entropy of vector to measure semantic richness"""
        # Discretize vector into bins for entropy calculation
        bins = np.linspace(vector.min(), vector.max(), 50)
        hist, _ = np.histogram(vector, bins=bins)
        hist = hist + 1e-10  # Avoid log(0)
        probs = hist / hist.sum()
        return entropy(probs, base=2)
    
    def _verify_geometric_properties(self, embeddings: np.ndarray) -> float:
        """Verify non-periodic tiling properties using geometric analysis"""
        if embeddings.shape[0] < 2:
            return 1.0
        
        # Compute pairwise cosine similarities
        similarities = np.zeros((embeddings.shape[0], embeddings.shape[0]))
        for i in range(embeddings.shape[0]):
            for j in range(i + 1, embeddings.shape[0]):
                sim = 1 - cosine(embeddings[i], embeddings[j])
                similarities[i, j] = similarities[j, i] = sim
        
        # Penrose tiling property: avoid periodic patterns
        # Check for excessive clustering or regular patterns
        mean_sim = np.mean(similarities[similarities > 0])
        std_sim = np.std(similarities[similarities > 0])
        
        # Good Penrose properties: moderate similarity with high variance
        # (avoiding both complete randomness and regular patterns)
        if mean_sim < 0.3:  # Too dispersed
            return 0.3
        elif mean_sim > 0.9:  # Too clustered
            return 0.2
        elif std_sim < 0.1:  # Too regular
            return 0.4
        else:
            # Score based on golden ratio relationships
            golden_score = self._check_golden_ratio_properties(similarities)
            return min(1.0, 0.7 + golden_score * 0.3)
    
    def _check_golden_ratio_properties(self, similarities: np.ndarray) -> float:
        """Check for golden ratio relationships in similarity distribution"""
        # Look for golden ratio proportions in similarity values
        unique_sims = np.unique(similarities[similarities > 0])
        if len(unique_sims) < 2:
            return 0.0
        
        ratios = []
        for i in range(len(unique_sims) - 1):
            ratio = unique_sims[i + 1] / unique_sims[i]
            ratios.append(ratio)
        
        # Check how close ratios are to golden ratio
        golden_distances = [abs(ratio - self.GOLDEN_RATIO) for ratio in ratios]
        if not golden_distances:
            return 0.0
        
        # Score based on proximity to golden ratio
        min_distance = min(golden_distances)
        return max(0.0, 1.0 - min_distance)
    
    def _compute_phase_coherence(self, embeddings: np.ndarray) -> float:
        """Compute phase coherence using oscillator synchronization metrics"""
        if embeddings.shape[0] < 2:
            return 1.0
        
        # Treat each embedding as a phase vector
        # Compute phase relationships between embeddings
        phases = np.angle(embeddings.view(complex).mean(axis=1))
        
        # Compute Kuramoto order parameter for phase synchronization
        complex_phases = np.exp(1j * phases)
        order_parameter = abs(np.mean(complex_phases))
        
        return float(order_parameter)
    
    def _estimate_semantic_stability(self, embeddings: np.ndarray, concepts: List[str]) -> float:
        """Estimate semantic stability (would require re-embedding in production)"""
        # For now, use intrinsic stability measures
        # In production, this would re-embed concepts and compare
        
        if embeddings.shape[0] < 2:
            return 1.0
        
        # Use vector consistency as proxy for stability
        centroid = np.mean(embeddings, axis=0)
        distances_to_centroid = [np.linalg.norm(emb - centroid) for emb in embeddings]
        stability = 1.0 - (np.std(distances_to_centroid) / np.mean(distances_to_centroid))
        
        return max(0.0, min(1.0, stability))
    
    def _generate_tessera_digest(self, embeddings: np.ndarray, concepts: List[str]) -> str:
        """Generate cryptographic digest for tessera integrity"""
        # Combine embeddings and concepts for digest
        embedding_bytes = embeddings.astype(np.float32).tobytes()
        concept_bytes = "::".join(concepts).encode('utf-8')
        
        combined = embedding_bytes + concept_bytes
        return hashlib.sha256(combined).hexdigest()

# Global verifier instance
_penrose_verifier = None

def get_penrose_verifier() -> EnhancedPenroseVerifier:
    """Get or create global Penrose verifier"""
    global _penrose_verifier
    if _penrose_verifier is None:
        _penrose_verifier = EnhancedPenroseVerifier()
    return _penrose_verifier
