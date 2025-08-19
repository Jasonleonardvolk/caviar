#!/usr/bin/env python3
"""
Modified origin_sentry.py - Updated to use TorusRegistry for No-DB persistence
Production-ready implementation with comprehensive error handling
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

# NEW IMPORTS for No-DB persistence
from python.core.torus_registry import TorusRegistry, get_torus_registry
from python.core.observer_synthesis import emit_token

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
    """
    Compatibility wrapper around TorusRegistry
    Maintains API compatibility while using Parquet-based persistence
    """
    
    def __init__(self, max_entries: int = 10000, max_size_mb: int = 200):
        """Initialize with TorusRegistry backend"""
        self.registry = get_torus_registry()
        self.max_entries = max_entries
        self.max_size_mb = max_size_mb
        
        # Cache for recent signatures (performance optimization)
        self._recent_cache = deque(maxlen=100)
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"SpectralDB initialized with TorusRegistry backend at {self.registry.path}")
        
    def add(self, signature: SpectralSignature):
        """Add new spectral signature using TorusRegistry"""
        try:
            # Record in registry
            shape_id = self.registry.record_shape(
                vertices=signature.eigenvalues,
                betti_numbers=signature.betti_numbers if signature.betti_numbers else [],
                coherence_band=signature.coherence_state,
                metadata={
                    'hash_id': signature.hash_id,
                    'gaps': signature.gaps,
                    'timestamp': signature.timestamp.isoformat()
                }
            )
            
            # Update cache
            self._recent_cache.append(signature)
            
            # Auto-flush periodically for durability
            if len(self._recent_cache) % 10 == 0:
                self.registry.flush()
                
            logger.debug(f"Added spectral signature {signature.hash_id} as {shape_id}")
            
        except Exception as e:
            logger.error(f"Failed to add spectral signature: {e}")
            raise
            
    def distance(self, eigenvalues: np.ndarray, top_k: int = 10) -> float:
        """
        Find minimum distance to any stored signature
        Uses cache first, then queries registry
        """
        if not eigenvalues.size:
            return float('inf')
            
        # Create temporary signature for comparison
        temp_sig = SpectralSignature(
            timestamp=datetime.now(timezone.utc),
            eigenvalues=eigenvalues,
            hash_id="temp"
        )
        
        # Check cache first
        cache_distances = []
        for sig in list(self._recent_cache)[-top_k:]:
            try:
                dist = temp_sig.distance_to(sig)
                cache_distances.append(dist)
            except Exception as e:
                logger.warning(f"Error computing distance in cache: {e}")
                
        if cache_distances:
            self._cache_hits += 1
            return min(cache_distances)
            
        # Fall back to registry query
        self._cache_misses += 1
        try:
            recent_df = self.registry.query_recent(top_k)
            if recent_df.empty:
                return float('inf')
                
            # Approximate distance using Betti numbers
            # This is a simplified metric - in production, you might store
            # compressed eigenvalue representations
            min_dist = float('inf')
            n_eigs = len(eigenvalues)
            
            for _, row in recent_df.iterrows():
                # Use Betti numbers as proxy for spectral structure
                b0_dist = abs(row['betti0'] - n_eigs)
                b1_dist = row['betti1'] * 0.1  # Weight topological complexity
                
                approx_dist = b0_dist + b1_dist
                min_dist = min(min_dist, approx_dist)
                
            return min_dist
            
        except Exception as e:
            logger.error(f"Failed to query registry for distances: {e}")
            return float('inf')
            
    def _save(self):
        """Persist to disk - delegates to registry"""
        try:
            self.registry.flush()
            logger.debug("SpectralDB flushed to registry")
        except Exception as e:
            logger.error(f"Failed to flush SpectralDB: {e}")
            
    def _load(self):
        """Load is handled automatically by TorusRegistry"""
        pass
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = self.registry.get_statistics()
        stats.update({
            'cache_size': len(self._recent_cache),
            'cache_hit_rate': self._cache_hits / max(1, self._cache_hits + self._cache_misses),
            'compatibility_mode': 'TorusRegistry'
        })
        return stats

class OriginSentry:
    """
    Evolution beyond EigenSentry - detects emergence of new cognitive dimensions
    Now with TorusRegistry persistence and observer token emission
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
            'spectral_entropy': 0.0,
            'tokens_emitted': 0
        }
        
        logger.info("OriginSentry initialized with TorusRegistry backend")
    
    def classify(self, eigenvalues: np.ndarray, betti_numbers: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Classify spectral state and detect emergent phenomena
        Now emits observer tokens for metacognitive feedback
        """
        # Validate input
        if not isinstance(eigenvalues, np.ndarray):
            eigenvalues = np.array(eigenvalues)
            
        # Sort eigenvalues by magnitude
        sorted_eigs = np.sort(np.abs(eigenvalues))[::-1]
        lambda_max = float(sorted_eigs[0]) if len(sorted_eigs) > 0 else 0.0
        
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
                    unseen_modes.append((i, float(eig)))
            
            self.metrics['dimension_expansions'] += 1
            self.dimension_births.append({
                'timestamp': datetime.now(timezone.utc),
                'new_modes': len(unseen_modes),
                'eigenvalues': [float(e) for _, e in unseen_modes[:5]]  # Limit for storage
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
            significant_gaps = [g for g in gaps if g > GAP_MIN]
            logger.info(f"🌈 Spectral gap birth detected! Gaps: {[f'{g:.3f}' for g in significant_gaps[:3]]}")
        
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
        coherence_transition = False
        if coherence != old_coherence:
            coherence_transition = True
            self.coherence_transitions.append({
                'timestamp': datetime.now(timezone.utc),
                'from': old_coherence,
                'to': coherence,
                'lambda_max': lambda_max
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
            'lambda_max': lambda_max
        })
        
        # Store signature
        self.spectral_db.add(signature)
        
        # 6. EMIT OBSERVER TOKEN for metacognitive feedback
        try:
            token = emit_token({
                "type": "origin_spectral",
                "source": "origin_sentry",
                "lambda_max": lambda_max,
                "coherence": coherence,
                "novelty_score": float(novelty_score),
                "dim_expansion": dim_expansion,
                "spectral_entropy": float(spectral_entropy),
                "gap_birth": gap_birth,
                "coherence_transition": coherence_transition,
                "unseen_modes": len(unseen_modes),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            self.metrics['tokens_emitted'] += 1
            logger.debug(f"Emitted observer token: {token[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to emit observer token: {e}")
        
        return {
            'dim_expansion': dim_expansion,
            'unseen_modes': len(unseen_modes),
            'gap_birth': gap_birth,
            'gaps': gaps,
            'coherence': coherence,
            'coherence_transition': coherence_transition,
            'novelty_score': float(novelty_score),
            'spectral_entropy': float(spectral_entropy),
            'lambda_max': lambda_max,
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
            try:
                # Ensure both lists have same length for comparison
                max_len = max(len(betti_numbers), len(self._last_betti))
                current_betti = list(betti_numbers) + [0.0] * (max_len - len(betti_numbers))
                last_betti = list(self._last_betti) + [0.0] * (max_len - len(self._last_betti))
                
                delta_betti = sum(abs(b1 - b2) for b1, b2 in zip(current_betti, last_betti))
                topo_novelty = min(delta_betti, 1.0)
            except Exception as e:
                logger.warning(f"Error computing topological novelty: {e}")
        
        if betti_numbers:
            self._last_betti = list(betti_numbers)
        
        return α * spectral_novelty + β * topo_novelty
    
    def _compute_spectral_entropy(self, eigenvalues: np.ndarray) -> float:
        """Compute Shannon entropy of spectral distribution"""
        # Normalize to probability distribution
        abs_eigs = np.abs(eigenvalues)
        total = np.sum(abs_eigs)
        
        if total == 0:
            return 0.0
        
        probs = abs_eigs / total
        
        # Remove zeros for log calculation
        probs = probs[probs > 0]
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log(probs))
        
        return float(entropy)
    
    def get_emergence_report(self) -> Dict[str, Any]:
        """Get comprehensive report on dimensional emergence"""
        # Get registry statistics
        registry_stats = self.spectral_db.get_statistics()
        
        return {
            'metrics': self.metrics.copy(),
            'dimension_births': self.dimension_births[-10:],  # Last 10
            'coherence_transitions': self.coherence_transitions[-10:],
            'novelty_trend': {
                'current': self.metrics['novelty_score'],
                'mean': float(np.mean(list(self.novelty_history))) if self.novelty_history else 0.0,
                'max': float(np.max(list(self.novelty_history))) if self.novelty_history else 0.0,
                'history_size': len(self.novelty_history)
            },
            'registry_stats': registry_stats,
            'tokens_emitted': self.metrics['tokens_emitted']
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
            logger.info(f"🎲 High novelty ({novelty:.3f}) - suggesting entropy injection")
            return True, 1.5
        elif novelty < threshold_low:
            # Low novelty - tighten control
            logger.info(f"📉 Low novelty ({novelty:.3f}) - suggesting control tightening")
            return True, 0.8
        else:
            # Moderate novelty - no change needed
            return False, 1.0
    
    def checkpoint(self) -> bool:
        """
        Create a checkpoint of current state
        Returns True if successful
        """
        try:
            # Flush registry to ensure persistence
            self.spectral_db._save()
            
            # Save additional metrics
            checkpoint_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'metrics': self.metrics,
                'dimension_births_count': len(self.dimension_births),
                'coherence_transitions_count': len(self.coherence_transitions),
                'novelty_history_size': len(self.novelty_history)
            }
            
            # Save checkpoint metadata
            checkpoint_path = Path(self.spectral_db.registry.path).parent / 'origin_sentry_checkpoint.json'
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
                
            logger.info(f"✅ Checkpoint created at {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return False

# Integration function for existing system
def evolve_eigensentry_to_origin():
    """
    Evolve existing EigenSentry to OriginSentry
    This function provides integration guidance
    """
    logger.info("🚀 Evolving EigenSentry → OriginSentry with TorusRegistry")
    
    # Create global OriginSentry instance
    origin_sentry = OriginSentry()
    
    # Integration code for eigensentry_guard.py
    integration_code = """
# Add to imports
from alan_backend.origin_sentry import OriginSentry

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
    
    return origin_sentry, integration_code

# Critic integration (if critic_hub available)
try:
    from kha.meta_genome.critics.critic_hub import critic

    @critic("stability_critic")
    def stability_critic(report: dict):
        λ = report.get("lambda_max", 0.0)
        score = max(0.0, 1.0 - λ / 0.05)
        return score, score >= 0.75
except ImportError:
    # Critic hub not available - skip registration
    pass

if __name__ == "__main__":
    # Test OriginSentry with TorusRegistry
    print("Testing OriginSentry with TorusRegistry backend...")
    
    # Ensure state directory exists
    import os
    if not os.getenv('TORI_STATE_ROOT'):
        os.environ['TORI_STATE_ROOT'] = './tori_state_test'
    
    origin = OriginSentry()
    
    # Simulate some eigenvalues
    test_eigenvalues = np.array([0.05, 0.03, 0.02, 0.01, 0.005])
    
    print("\nTest 1: Basic classification")
    result = origin.classify(test_eigenvalues)
    print(f"Classification result: {json.dumps(result, indent=2)}")
    
    print("\nTest 2: With Betti numbers")
    result2 = origin.classify(test_eigenvalues, betti_numbers=[1.0, 0.0, 0.0])
    print(f"With Betti: novelty_score={result2['novelty_score']:.3f}")
    
    # Check entropy injection
    print("\nTest 3: Entropy injection recommendation")
    should_inject, factor = origin.should_inject_entropy()
    print(f"Entropy injection: {should_inject}, factor: {factor}")
    
    # Get emergence report
    print("\nTest 4: Emergence report")
    report = origin.get_emergence_report()
    print(f"Report summary: {report['metrics']['tokens_emitted']} tokens emitted")
    print(f"Registry stats: {report['registry_stats']}")
    
    # Create checkpoint
    print("\nTest 5: Creating checkpoint")
    success = origin.checkpoint()
    print(f"Checkpoint created: {success}")
    
    print("\n✅ All tests completed!")
