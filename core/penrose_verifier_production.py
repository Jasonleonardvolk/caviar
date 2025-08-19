# core/penrose_verifier_production.py - Fixed complex casting and entropy
import numpy as np
import logging
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass
import hashlib
import os
from scipy.spatial.distance import cosine
from sklearn.neighbors import KernelDensity

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

class ProductionPenroseVerifier:
    """Production Penrose verifier with fixed math and configurable thresholds"""
    
    def __init__(self):
        # Configurable thresholds via environment variables
        self.COSINE_THRESHOLD = float(os.getenv("TORI_COSINE_THRESHOLD", "0.65"))
        self.VECTOR_NORM_MIN = float(os.getenv("TORI_NORM_MIN", "0.9"))
        self.VECTOR_NORM_MAX = float(os.getenv("TORI_NORM_MAX", "1.1"))
        self.ENTROPY_THRESHOLD = float(os.getenv("TORI_ENTROPY_THRESHOLD", "4.0"))
        self.STABILITY_THRESHOLD = float(os.getenv("TORI_STABILITY_THRESHOLD", "0.92"))
        self.PENROSE_PASS_RATE_THRESHOLD = float(os.getenv("TORI_PENROSE_PASS_RATE", "0.95"))
        self.GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
        
        logger.info(f"Production Penrose Verifier initialized with thresholds:")
        logger.info(f"  Cosine: {self.COSINE_THRESHOLD}")
        logger.info(f"  Norm: [{self.VECTOR_NORM_MIN}, {self.VECTOR_NORM_MAX}]")
        logger.info(f"  Entropy: {self.ENTROPY_THRESHOLD}")
        logger.info(f"  Stability: {self.STABILITY_THRESHOLD}")
    
    def verify_tessera(
        self,
        embeddings: np.ndarray,
        concepts: List[str],
        metadata: Dict[str, Any] = None
    ) -> PenroseVerificationResult:
        """Comprehensive tessera verification with fixed math"""
        
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
        
        # Ensure embeddings are unit normalized (assumption for thresholds)
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=0.1):
            logger.warning("Embeddings are not unit normalized, normalizing now")
            embeddings = embeddings / norms[:, np.newaxis]
        
        # Quality Gate 1: Vector norm validation
        norm_check = np.all((norms >= self.VECTOR_NORM_MIN) & (norms <= self.VECTOR_NORM_MAX))
        
        # Quality Gate 2: Fixed entropy validation using kernel density estimation
        entropies = [self._calculate_kde_entropy(emb) for emb in embeddings]
        entropy_check = np.all(np.array(entropies) > self.ENTROPY_THRESHOLD)
        
        # Quality Gate 3: Geometric verification
        geometric_score = self._verify_geometric_properties(embeddings)
        geometric_check = geometric_score > 0.7
        
        # Quality Gate 4: Fixed phase coherence without complex casting issues
        phase_coherence = self._compute_phase_coherence_fixed(embeddings)
        phase_check = phase_coherence > 0.6
        
        # Quality Gate 5: Semantic stability
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
        
        # Calculate pass rate for SLO monitoring
        passed_gates = sum(vector_quality.values())
        pass_rate = passed_gates / len(vector_quality)
        
        result_metadata = {
            "vector_count": embeddings.shape[0],
            "vector_dim": embeddings.shape[1],
            "mean_norm": float(np.mean(norms)),
            "mean_entropy": float(np.mean(entropies)),
            "pass_rate": pass_rate,
            "slo_met": pass_rate >= self.PENROSE_PASS_RATE_THRESHOLD,
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
        
        logger.info(f"Penrose verification: {status} (pass_rate={pass_rate:.3f}, SLO_met={result_metadata['slo_met']})")
        
        if status == "FAILED":
            failed_checks = [k for k, v in vector_quality.items() if not v]
            raise ValueError(f"Penrose verification FAILED. Failed checks: {failed_checks}")
        
        return verification_result
    
    def _calculate_kde_entropy(self, vector: np.ndarray) -> float:
        """Fixed entropy calculation using kernel density estimation"""
        # Reshape for sklearn
        vector_reshaped = vector.reshape(-1, 1)
        
        # Use kernel density estimation for better entropy estimation
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
        kde.fit(vector_reshaped)
        
        # Sample points for entropy calculation
        sample_points = np.linspace(vector.min(), vector.max(), 100).reshape(-1, 1)
        log_density = kde.score_samples(sample_points)
        
        # Calculate entropy: H = -∫ p(x) log p(x) dx
        # Approximate with discrete sum
        density = np.exp(log_density)
        density = density / np.sum(density)  # Normalize
        
        # Avoid log(0)
        density = density + 1e-10
        entropy = -np.sum(density * np.log2(density))
        
        return float(entropy)
    
    def _compute_phase_coherence_fixed(self, embeddings: np.ndarray) -> float:
        """Fixed phase coherence without complex casting issues"""
        if embeddings.shape[0] < 2:
            return 1.0
        
        # Use angle between consecutive embedding vectors as phase
        phases = []
        for i in range(len(embeddings) - 1):
            # Compute angle between consecutive vectors
            cos_angle = np.dot(embeddings[i], embeddings[i + 1])
            # Clamp to valid range for arccos
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            phases.append(angle)
        
        phases = np.array(phases)
        
        # Compute order parameter for phase synchronization
        # Use complex exponentials without view casting
        complex_phases = np.cos(phases) + 1j * np.sin(phases)
        order_parameter = abs(np.mean(complex_phases))
        
        return float(order_parameter)
    
    def _verify_geometric_properties(self, embeddings: np.ndarray) -> float:
        """Verify non-periodic tiling properties"""
        if embeddings.shape[0] < 2:
            return 1.0
        
        # Compute pairwise cosine similarities efficiently
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        
        # Remove diagonal elements
        mask = ~np.eye(similarities.shape[0], dtype=bool)
        sim_values = similarities[mask]
        
        mean_sim = np.mean(sim_values)
        std_sim = np.std(sim_values)
        
        # Penrose tiling properties: moderate similarity with high variance
        if mean_sim < 0.3:  # Too dispersed
            return 0.3
        elif mean_sim > 0.9:  # Too clustered
            return 0.2
        elif std_sim < 0.1:  # Too regular
            return 0.4
        else:
            # Score based on golden ratio relationships
            golden_score = self._check_golden_ratio_properties(sim_values)
            return min(1.0, 0.7 + golden_score * 0.3)
    
    def _check_golden_ratio_properties(self, similarities: np.ndarray) -> float:
        """Check for golden ratio relationships in similarity distribution"""
        unique_sims = np.unique(similarities[similarities > 0])
        if len(unique_sims) < 2:
            return 0.0
        
        ratios = []
        for i in range(len(unique_sims) - 1):
            if unique_sims[i] > 0:  # Avoid division by zero
                ratio = unique_sims[i + 1] / unique_sims[i]
                ratios.append(ratio)
        
        if not ratios:
            return 0.0
        
        # Check how close ratios are to golden ratio
        golden_distances = [abs(ratio - self.GOLDEN_RATIO) for ratio in ratios]
        min_distance = min(golden_distances)
        
        # Score based on proximity to golden ratio
        return max(0.0, 1.0 - min_distance)
    
    def _estimate_semantic_stability(self, embeddings: np.ndarray, concepts: List[str]) -> float:
        """Estimate semantic stability using intrinsic measures"""
        if embeddings.shape[0] < 2:
            return 1.0
        
        # Use vector consistency as proxy for stability
        centroid = np.mean(embeddings, axis=0)
        distances_to_centroid = [np.linalg.norm(emb - centroid) for emb in embeddings]
        
        if np.mean(distances_to_centroid) == 0:
            return 1.0
        
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

def get_penrose_verifier() -> ProductionPenroseVerifier:
    """Get or create global Penrose verifier"""
    global _penrose_verifier
    if _penrose_verifier is None:
        _penrose_verifier = ProductionPenroseVerifier()
    return _penrose_verifier
