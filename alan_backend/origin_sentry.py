#!/usr/bin/env python3
"""
OriginSentry - Beyond Metacognition to Self-Transforming Cognition
Detects spectral growth, gap births, and coherence transitions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from collections import deque
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

# Constants for spectral analysis
EPS = 0.01  # Novelty threshold
GAP_MIN = 0.02  # Minimum gap for birth detection
COHERENCE_BANDS = {
    'local': (0, 0.01),
    'global': (0.01, 0.04),
    'critical': (0.04, float('inf'))
}

@dataclass
class SpectralSignature:
    """Represents a spectral fingerprint at a moment in time"""
    timestamp: datetime
    eigenvalues: np.ndarray
    hash_id: str
    betti_numbers: Optional[List[float]] = None
    coherence_state: str = 'local'
    gaps: List[float] = field(default_factory=list)
    
    def distance_to(self, other: 'SpectralSignature') -> float:
        """Compute spectral distance using Wasserstein metric"""
        # Sort eigenvalues for proper comparison
        sorted_self = np.sort(np.abs(self.eigenvalues))
        sorted_other = np.sort(np.abs(other.eigenvalues))
        
        # Pad shorter array with zeros
        max_len = max(len(sorted_self), len(sorted_other))
        padded_self = np.pad(sorted_self, (0, max_len - len(sorted_self)))
        padded_other = np.pad(sorted_other, (0, max_len - len(sorted_other)))
        
        # Wasserstein-like distance
        return np.mean(np.abs(padded_self - padded_other))

class SpectralDB:
    """Lightweight database for spectral history with LRU eviction"""
    
    def __init__(self, max_entries: int = 10000, max_size_mb: int = 200):
        self.max_entries = max_entries
        self.max_size_mb = max_size_mb
        self.signatures = deque(maxlen=max_entries)
        self.hash_index = {}
        self.storage_path = Path("spectral_db.json")
        self._load()
    
    def add(self, signature: SpectralSignature):
        """Add new spectral signature with LRU eviction"""
        # Check size limit
        if self._estimate_size_mb() > self.max_size_mb and len(self.signatures) > 0:
            # Evict oldest
            oldest = self.signatures.popleft()
            del self.hash_index[oldest.hash_id]
            logger.debug(f"Evicted old spectral signature: {oldest.hash_id}")
        
        self.signatures.append(signature)
        self.hash_index[signature.hash_id] = signature
        self._save()
    
    def distance(self, eigenvalues: np.ndarray, top_k: int = 10) -> float:
        """Find minimum distance to any stored signature"""
        if not self.signatures:
            return float('inf')
        
        # Create temporary signature for comparison
        temp_sig = SpectralSignature(
            timestamp=datetime.now(timezone.utc),
            eigenvalues=eigenvalues,
            hash_id="temp"
        )
        
        # Check against recent signatures (more efficient)
        distances = []
        for sig in list(self.signatures)[-top_k:]:
            distances.append(temp_sig.distance_to(sig))
        
        return min(distances) if distances else float('inf')
    
    def _estimate_size_mb(self) -> float:
        """Estimate current database size in MB"""
        # Rough estimate: each signature ~1KB
        return len(self.signatures) * 0.001
    
    def _save(self):
        """Persist to disk"""
        try:
            data = []
            for sig in list(self.signatures)[-100:]:  # Save only recent 100
                data.append({
                    'timestamp': sig.timestamp.isoformat(),
                    'eigenvalues': sig.eigenvalues.tolist(),
                    'hash_id': sig.hash_id,
                    'coherence_state': sig.coherence_state,
                    'gaps': sig.gaps
                })
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save spectral DB: {e}")
    
    def _load(self):
        """Load from disk"""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            for item in data:
                sig = SpectralSignature(
                    timestamp=datetime.fromisoformat(item['timestamp']),
                    eigenvalues=np.array(item['eigenvalues']),
                    hash_id=item['hash_id'],
                    coherence_state=item.get('coherence_state', 'local'),
                    gaps=item.get('gaps', [])
                )
                self.signatures.append(sig)
                self.hash_index[sig.hash_id] = sig
                
        except Exception as e:
            logger.error(f"Failed to load spectral DB: {e}")

class OriginSentry:
    """
    Evolution beyond EigenSentry - detects emergence of new cognitive dimensions
    """
    
    def __init__(self, spectral_db: Optional[SpectralDB] = None):
        self.spectral_db = spectral_db or SpectralDB()
        self.novelty_history = deque(maxlen=1000)
        self.dimension_births = []
        self.coherence_transitions = []
        
        # Metrics
        self.metrics = {
            'current_dimension': 0,
            'dimension_expansions': 0,
            'gap_births': 0,
            'coherence_state': 'local',
            'novelty_score': 0.0,
            'spectral_entropy': 0.0
        }
        
        logger.info("OriginSentry initialized - monitoring for dimensional emergence")
    
    def classify(self, eigenvalues: np.ndarray, betti_numbers: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Classify spectral state and detect emergent phenomena
        """
        # Sort eigenvalues by magnitude
        sorted_eigs = np.sort(np.abs(eigenvalues))[::-1]
        lambda_max = sorted_eigs[0] if len(sorted_eigs) > 0 else 0.0
        
        # Create signature
        hash_id = hashlib.md5(eigenvalues.tobytes()).hexdigest()[:8]
        signature = SpectralSignature(
            timestamp=datetime.now(timezone.utc),
            eigenvalues=eigenvalues,
            hash_id=hash_id,
            betti_numbers=betti_numbers
        )
        
        # 1. Detect unseen spectral modes (dimension expansion)
        min_distance = self.spectral_db.distance(eigenvalues)
        dim_expansion = min_distance > EPS
        unseen_modes = []
        
        if dim_expansion:
            # Find which modes are new
            for i, eig in enumerate(eigenvalues):
                mode_distance = self.spectral_db.distance(np.array([eig]))
                if mode_distance > EPS:
                    unseen_modes.append((i, eig))
            
            self.metrics['dimension_expansions'] += 1
            self.dimension_births.append({
                'timestamp': datetime.now(timezone.utc),
                'new_modes': len(unseen_modes),
                'eigenvalues': [float(e) for _, e in unseen_modes]
            })
            logger.info(f"🌟 Dimensional expansion detected! {len(unseen_modes)} new modes")
        
        # 2. Detect spectral gap births
        gaps = []
        if len(sorted_eigs) > 1:
            for i in range(len(sorted_eigs) - 1):
                gap = abs(sorted_eigs[i] - sorted_eigs[i+1])
                gaps.append(float(gap))
        
        gap_birth = any(g > GAP_MIN for g in gaps)
        if gap_birth:
            self.metrics['gap_births'] += 1
            logger.info(f"🌈 Spectral gap birth detected! Gaps: {[f'{g:.3f}' for g in gaps if g > GAP_MIN]}")
        
        signature.gaps = gaps
        
        # 3. Determine coherence state
        old_coherence = self.metrics['coherence_state']
        
        if lambda_max > COHERENCE_BANDS['critical'][0]:
            coherence = 'critical'
        elif lambda_max > COHERENCE_BANDS['global'][0]:
            coherence = 'global'
        else:
            coherence = 'local'
        
        signature.coherence_state = coherence
        
        # Detect coherence transitions
        if coherence != old_coherence:
            self.coherence_transitions.append({
                'timestamp': datetime.now(timezone.utc),
                'from': old_coherence,
                'to': coherence,
                'lambda_max': float(lambda_max)
            })
            logger.info(f"🔄 Coherence transition: {old_coherence} → {coherence}")
        
        # 4. Calculate novelty score
        novelty_score = self._compute_novelty(eigenvalues, betti_numbers, min_distance)
        self.novelty_history.append(novelty_score)
        
        # 5. Calculate spectral entropy
        spectral_entropy = self._compute_spectral_entropy(sorted_eigs)
        
        # Update metrics
        self.metrics.update({
            'current_dimension': len(eigenvalues),
            'coherence_state': coherence,
            'novelty_score': float(novelty_score),
            'spectral_entropy': float(spectral_entropy),
            'lambda_max': float(lambda_max)
        })
        
        # Store signature
        self.spectral_db.add(signature)
        
        return {
            'dim_expansion': dim_expansion,
            'unseen_modes': len(unseen_modes),
            'gap_birth': gap_birth,
            'gaps': gaps,
            'coherence': coherence,
            'coherence_transition': coherence != old_coherence,
            'novelty_score': float(novelty_score),
            'spectral_entropy': float(spectral_entropy),
            'lambda_max': float(lambda_max),
            'metrics': self.metrics.copy()
        }
    
    def _compute_novelty(self, eigenvalues: np.ndarray, 
                        betti_numbers: Optional[List[float]], 
                        spectral_distance: float) -> float:
        """
        Compute novelty score combining spectral and topological features
        N_t = α * JSdiv(p(ω_t) || p(ω_{t-1})) + β * ΔBetti
        """
        α = 0.7  # Weight for spectral novelty
        β = 0.3  # Weight for topological novelty
        
        # Spectral component (using distance as proxy for JS divergence)
        spectral_novelty = min(spectral_distance / EPS, 1.0)
        
        # Topological component
        topo_novelty = 0.0
        if betti_numbers and hasattr(self, '_last_betti'):
            delta_betti = sum(abs(b1 - b2) for b1, b2 in 
                            zip(betti_numbers, self._last_betti))
            topo_novelty = min(delta_betti, 1.0)
        
        if betti_numbers:
            self._last_betti = betti_numbers
        
        return α * spectral_novelty + β * topo_novelty
    
    def _compute_spectral_entropy(self, eigenvalues: np.ndarray) -> float:
        """Compute Shannon entropy of spectral distribution"""
        # Normalize to probability distribution
        abs_eigs = np.abs(eigenvalues)
        if np.sum(abs_eigs) == 0:
            return 0.0
        
        probs = abs_eigs / np.sum(abs_eigs)
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return float(entropy)
    
    def get_emergence_report(self) -> Dict[str, Any]:
        """Get comprehensive report on dimensional emergence"""
        return {
            'metrics': self.metrics.copy(),
            'dimension_births': self.dimension_births[-10:],  # Last 10
            'coherence_transitions': self.coherence_transitions[-10:],
            'novelty_trend': {
                'current': self.metrics['novelty_score'],
                'mean': float(np.mean(list(self.novelty_history))) if self.novelty_history else 0.0,
                'max': float(np.max(list(self.novelty_history))) if self.novelty_history else 0.0
            },
            'spectral_db_size': len(self.spectral_db.signatures)
        }
    
    def should_inject_entropy(self, threshold_high: float = 0.7, 
                            threshold_low: float = 0.2) -> Tuple[bool, float]:
        """
        Determine if entropy injection is needed for creative exploration
        Returns (should_inject, suggested_lambda_factor)
        """
        novelty = self.metrics['novelty_score']
        
        if novelty > threshold_high:
            # High novelty - increase chaos for exploration
            return True, 1.5
        elif novelty < threshold_low:
            # Low novelty - tighten control
            return True, 0.8
        else:
            # Moderate novelty - no change needed
            return False, 1.0

# Integration function for existing system
def evolve_eigensentry_to_origin():
    """
    Evolve existing EigenSentry to OriginSentry
    This function patches the existing system
    """
    logger.info("🚀 Evolving EigenSentry → OriginSentry")
    
    # Create global OriginSentry instance
    origin_sentry = OriginSentry()
    
    # Patch into existing eigensentry_guard.py
    patch_code = """
# Add to imports
from alan_backend.origin_sentry import OriginSentry, evolve_eigensentry_to_origin

# In CurvatureAwareGuard.__init__, add:
        self.origin_sentry = OriginSentry()
        
# In check_eigenvalues method, after computing eigenvalues, add:
        # Classify with OriginSentry
        origin_classification = self.origin_sentry.classify(eigenvalues)
        
        # Check for entropy injection
        should_inject, lambda_factor = self.origin_sentry.should_inject_entropy()
        if should_inject:
            logger.info(f"🎲 Entropy injection: lambda_factor={lambda_factor}")
            # Modify threshold based on novelty
            self.current_threshold *= lambda_factor
"""
    
    return origin_sentry, patch_code

# append near bottom
from kha.meta_genome.critics.critic_hub import critic

@critic("stability_critic")
def stability_critic(report: dict):
    λ = report.get("lambda_max", 0.0)
    score = max(0.0, 1.0 - λ / 0.05)
    return score, score >= 0.75

if __name__ == "__main__":
    # Test OriginSentry
    origin = OriginSentry()
    
    # Simulate some eigenvalues
    test_eigenvalues = np.array([0.05, 0.03, 0.02, 0.01, 0.005])
    
    result = origin.classify(test_eigenvalues)
    print("Classification result:", json.dumps(result, indent=2))
    
    # Check entropy injection
    should_inject, factor = origin.should_inject_entropy()
    print(f"\nEntropy injection: {should_inject}, factor: {factor}")
    
    # Get emergence report
    report = origin.get_emergence_report()
    print("\nEmergence report:", json.dumps(report, indent=2))
