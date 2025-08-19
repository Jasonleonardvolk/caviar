"""
ALAN integration bridge for ψ-Sync stability monitoring.

This module connects the PsiSyncMonitor to ALAN's core orchestration system,
enabling stability-aware cognitive processing. It provides methods for:

1. Checking concept stability during reasoning steps
2. Applying corrective adjustments to phase oscillator dynamics
3. Making confidence-weighted decisions based on stability metrics
4. Feeding back spectral insights to the concept network

This bridge allows ALAN to autonomously monitor and maintain cognitive stability
while processing complex information.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from enum import Enum
import json
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("alan_psi_sync_bridge")

# Import our modules
from alan_backend.banksy import (
    PsiSyncMonitor, 
    PsiPhaseState, 
    PsiSyncMetrics, 
    SyncAction,
    SyncState,
    get_psi_sync_monitor
)

class AlanPsiState(Enum):
    """ALAN-specific psi-phase stability states."""
    COHERENT = "coherent"      # Stable, high confidence processing
    UNCERTAIN = "uncertain"    # Moderate drift, proceed with caution
    INCOHERENT = "incoherent"  # Unstable, needs user clarification
    UNINITIALIZED = "uninitialized"  # No stability data yet

class AlanPsiSyncBridge:
    """
    Bridge between ALAN's orchestration system and the ψ-Sync stability monitoring.
    
    This class provides the integration layer between ALAN's core cognitive
    architecture and the phase-eigenfunction synchronization monitoring system.
    It translates low-level stability metrics into actionable decisions for
    ALAN's reasoning and response generation.
    
    Attributes:
        monitor: The PsiSyncMonitor instance for stability assessment
        stable_threshold: Threshold for coherent state
        drift_threshold: Threshold for uncertain state
        last_state: The last processed state
        active_concepts: Currently active concepts in the system
        stability_history: Recent stability assessments
    """
    
    def __init__(
        self,
        stable_threshold: float = 0.9,
        drift_threshold: float = 0.6,
    ):
        """
        Initialize the AlanPsiSyncBridge.
        
        Args:
            stable_threshold: Threshold for coherent state
            drift_threshold: Threshold for uncertain state
        """
        # Get or create monitor
        self.monitor = get_psi_sync_monitor(
            stable_threshold=stable_threshold,
            drift_threshold=drift_threshold
        )
        
        # Store thresholds
        self.stable_threshold = stable_threshold
        self.drift_threshold = drift_threshold
        
        # Track state
        self.last_state: Optional[PsiPhaseState] = None
        self.last_metrics: Optional[PsiSyncMetrics] = None
        self.active_concepts: Set[str] = set()
        
        # History for trend analysis
        self.stability_history: List[Tuple[float, float]] = []  # (synchrony, integrity)
        
        logger.info(
            f"AlanPsiSyncBridge initialized with thresholds: "
            f"stable={stable_threshold}, drift={drift_threshold}"
        )
    
    def check_concept_stability(
        self,
        phases: np.ndarray,
        psi_values: np.ndarray,
        concept_ids: List[str],
        coupling_matrix: Optional[np.ndarray] = None
    ) -> Tuple[AlanPsiState, float, str]:
        """
        Check stability of the given concept state.
        
        This method assesses the stability of the current concept network state
        and returns ALAN-specific stability information for decision-making.
        
        Args:
            phases: Current phase values for oscillators
            psi_values: Current eigenfunction values (or vectors)
            concept_ids: Identifiers for the concepts
            coupling_matrix: Optional coupling matrix between oscillators
            
        Returns:
            Tuple of (stability_state, confidence, recommendation)
        """
        # Create state object
        state = PsiPhaseState(
            theta=phases,
            psi=psi_values,
            coupling_matrix=coupling_matrix,
            concept_ids=concept_ids
        )
        
        # Evaluate stability
        metrics = self.monitor.evaluate(state)
        
        # Map to ALAN-specific state
        if metrics.sync_state == SyncState.STABLE:
            alan_state = AlanPsiState.COHERENT
        elif metrics.sync_state == SyncState.DRIFT:
            alan_state = AlanPsiState.UNCERTAIN
        else:  # BREAK or UNKNOWN
            alan_state = AlanPsiState.INCOHERENT
            
        # Get recommendations
        action = self.monitor.recommend_action(metrics, state)
        
        # Store state for history
        self.last_state = state
        self.last_metrics = metrics
        self.active_concepts = set(concept_ids)
        
        # Update stability history
        self.stability_history.append((metrics.synchrony_score, metrics.attractor_integrity))
        if len(self.stability_history) > 10:  # Keep limited history
            self.stability_history.pop(0)
            
        logger.info(
            f"Concept stability: {alan_state.value}, "
            f"synchrony={metrics.synchrony_score:.2f}, confidence={action.confidence:.2f}"
        )
        
        return alan_state, action.confidence, action.recommendation
    
    def get_coupling_adjustments(self) -> Optional[Dict[Tuple[str, str], float]]:
        """
        Get recommended coupling adjustments between concepts.
        
        Returns:
            Dictionary mapping concept pairs to adjustment values,
            or None if no adjustments available
        """
        if self.last_state is None or self.last_metrics is None:
            return None
            
        # Get action
        action = self.monitor.recommend_action(self.last_metrics, self.last_state)
        
        if action.coupling_adjustments is None:
            return None
            
        # Convert matrix to concept pair mappings
        adjustments: Dict[Tuple[str, str], float] = {}
        
        for i, concept_i in enumerate(self.last_state.concept_ids):
            for j, concept_j in enumerate(self.last_state.concept_ids):
                if i != j and abs(action.coupling_adjustments[i, j]) > 0.01:  # Threshold for significance
                    adjustments[(concept_i, concept_j)] = float(action.coupling_adjustments[i, j])
                    
        return adjustments
    
    def should_request_clarification(self) -> bool:
        """
        Determine if the system should request user clarification.
        
        Returns:
            True if clarification is recommended based on stability
        """
        if self.last_metrics is None:
            return False
            
        # Check if state is unstable
        is_unstable = self.last_metrics.sync_state == SyncState.BREAK
        
        # Also check for consistent drift
        drift_trend = False
        if len(self.stability_history) >= 3:
            # Check if synchrony has been consistently low
            recent_scores = [s[0] for s in self.stability_history[-3:]]
            if all(s < self.stable_threshold for s in recent_scores):
                drift_trend = True
                
        return is_unstable or drift_trend
    
    def estimate_response_confidence(self) -> float:
        """
        Estimate confidence level for ALAN's responses based on stability.
        
        Returns:
            Confidence score (0-1) for weighting responses
        """
        if self.last_metrics is None:
            return 0.5  # Default moderate confidence
            
        # Compute confidence based on synchrony and integrity
        confidence = 0.5 * (self.last_metrics.synchrony_score + self.last_metrics.attractor_integrity)
        
        # Reduce confidence if residual energy is high
        if self.last_metrics.residual_energy > 0.3:
            confidence *= (1.0 - self.last_metrics.residual_energy)
            
        # Ensure in range [0,1]
        return max(0.0, min(1.0, confidence))
    
    def get_stability_report(self) -> Dict[str, Any]:
        """
        Get a detailed stability report for diagnostics.
        
        Returns:
            Dictionary with stability metrics and assessments
        """
        if self.last_metrics is None:
            return {
                "status": "uninitialized",
                "message": "No stability data available yet"
            }
            
        # Map sync state to ALAN state
        state_map = {
            SyncState.STABLE: AlanPsiState.COHERENT,
            SyncState.DRIFT: AlanPsiState.UNCERTAIN,
            SyncState.BREAK: AlanPsiState.INCOHERENT,
            SyncState.UNKNOWN: AlanPsiState.UNINITIALIZED
        }
        
        alan_state = state_map[self.last_metrics.sync_state]
        
        # Get action
        action = self.monitor.recommend_action(self.last_metrics, self.last_state)
        
        # Build report
        report = {
            "status": alan_state.value,
            "metrics": {
                "synchrony_score": float(self.last_metrics.synchrony_score),
                "attractor_integrity": float(self.last_metrics.attractor_integrity),
                "residual_energy": float(self.last_metrics.residual_energy),
                "lyapunov_delta": float(self.last_metrics.lyapunov_delta),
                "active_modes": list(map(int, self.last_metrics.active_mode_indices))
            },
            "assessment": {
                "confidence": float(action.confidence),
                "recommendation": action.recommendation,
                "requires_confirmation": action.requires_user_confirmation
            },
            "active_concepts": list(self.active_concepts)
        }
        
        # Add coupling info if available
        if action.coupling_adjustments is not None:
            total_adjustment = float(np.sum(np.abs(action.coupling_adjustments)))
            max_adjustment = float(np.max(np.abs(action.coupling_adjustments)))
            
            report["coupling"] = {
                "total_adjustment": total_adjustment,
                "max_adjustment": max_adjustment,
                "significant_adjustments": len(self.get_coupling_adjustments() or {})
            }
            
        return report

def get_alan_psi_bridge() -> AlanPsiSyncBridge:
    """
    Get or create the ALAN ψ-Sync bridge instance.
    
    This function provides a singleton-like access to the bridge,
    ensuring consistent stability monitoring across the system.
    
    Returns:
        AlanPsiSyncBridge instance
    """
    # This could be enhanced to maintain a true singleton
    return AlanPsiSyncBridge(
        stable_threshold=0.85,  # Slightly reduced from default
        drift_threshold=0.6
    )

# Example usage
if __name__ == "__main__":
    # Create bridge
    bridge = get_alan_psi_bridge()
    
    # Generate sample data
    n_concepts = 5
    concept_ids = ["Memory", "Attention", "Reasoning", "Perception", "Learning"]
    
    # Coherent case
    mean_phase = np.random.uniform(0, 2*np.pi)
    phases_coherent = mean_phase + np.random.normal(0, 0.2, n_concepts)
    
    # Incoherent case
    phases_incoherent = np.random.uniform(0, 2*np.pi, n_concepts)
    
    # Simple eigenfunction values
    psi_values = np.random.normal(0, 1, (n_concepts, 2))
    
    # Check coherent case
    print("\n=== TESTING COHERENT CASE ===")
    state, confidence, recommendation = bridge.check_concept_stability(
        phases=phases_coherent,
        psi_values=psi_values,
        concept_ids=concept_ids
    )
    
    print(f"Stability state: {state.value}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Recommendation: {recommendation}")
    print(f"Should request clarification: {bridge.should_request_clarification()}")
    
    # Check incoherent case
    print("\n=== TESTING INCOHERENT CASE ===")
    state, confidence, recommendation = bridge.check_concept_stability(
        phases=phases_incoherent,
        psi_values=psi_values,
        concept_ids=concept_ids
    )
    
    print(f"Stability state: {state.value}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Recommendation: {recommendation}")
    print(f"Should request clarification: {bridge.should_request_clarification()}")
    
    # Print full report
    print("\n=== STABILITY REPORT ===")
    report = bridge.get_stability_report()
    print(json.dumps(report, indent=2))
