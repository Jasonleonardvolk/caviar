"""
ψ-Concept Bridge for ELFIN and PsiSyncMonitor Integration.

This module provides the critical bridge between the ELFIN DSL's concept network
and the ψ-Sync monitor's phase-space representation. It enables:

1. Bidirectional mapping between concepts and phase oscillators
2. Translation of Lyapunov stability constraints to ψ-space
3. Monitoring of phase drift for concept stability
4. Adaptive feedback for oscillator coupling adjustment

This bridge enables the ELFIN DSL to formally reason about and enforce
stability properties in the phase-coupled concept system.
"""

import logging
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable
from dataclasses import dataclass, field

# Import ELFIN components
from alan_backend.elfin.stability.lyapunov import (
    LyapunovFunction,
    ProofStatus,
    PolynomialLyapunov,
    NeuralLyapunov,
    CLVFunction,
    CompositeLyapunov
)

# Import ψ-Sync components
from alan_backend.banksy import (
    PsiSyncMonitor,
    PsiPhaseState,
    PsiSyncMetrics,
    SyncAction,
    SyncState
)

# Configure logger
logger = logging.getLogger("elfin.stability.psi_bridge")

@dataclass
class PhaseStateUpdate:
    """
    Update notification for phase state changes.
    
    This represents a change in the phase-space state that
    should be propagated to the concept network.
    
    Attributes:
        concept_ids: List of concept IDs affected by the update
        theta_updates: Phase updates for each concept
        psi_updates: Eigenfunction updates for each concept
        sync_metrics: Current synchronization metrics
    """
    
    concept_ids: List[str]
    theta_updates: np.ndarray
    psi_updates: np.ndarray
    sync_metrics: PsiSyncMetrics
    
    def __post_init__(self):
        """Validate the update data."""
        # Ensure dimensions match
        assert len(self.concept_ids) == len(self.theta_updates), "Concept IDs and theta updates must have same length"
        assert len(self.concept_ids) == self.psi_updates.shape[0], "Concept IDs and psi updates must have same length"


@dataclass
class ConceptPhaseMapping:
    """
    Mapping between a concept and its phase representation.
    
    This defines how a concept in the ELFIN LocalConceptNetwork
    maps to oscillators and eigenfunctions in the phase-space.
    
    Attributes:
        concept_id: Identifier for the concept
        phase_index: Index in the phase oscillator array
        psi_mode_indices: Indices of ψ-modes that this concept participates in
        psi_mode_weights: Weights for each ψ-mode
        lyapunov_functions: Lyapunov functions that involve this concept
    """
    
    concept_id: str
    phase_index: int
    psi_mode_indices: List[int]
    psi_mode_weights: List[float] = field(default_factory=lambda: [1.0])
    lyapunov_functions: List[str] = field(default_factory=list)


class PsiConceptBridge:
    """
    Bridge between ELFIN concept network and ψ-Sync phase state.
    
    This class provides the critical interface between the symbolic
    reasoning of ELFIN and the dynamical system monitored by the
    ψ-Sync monitor. It handles the bidirectional mapping between
    concepts and phase oscillators, and provides stability monitoring
    and enforcement based on Lyapunov functions.
    """
    
    def __init__(
        self,
        psi_sync_monitor: PsiSyncMonitor,
        synchrony_threshold: float = 0.85,
        update_callback: Optional[Callable[[PhaseStateUpdate], None]] = None
    ):
        """
        Initialize the bridge.
        
        Args:
            psi_sync_monitor: Monitor for phase-eigenfunction synchronization
            synchrony_threshold: Threshold for synchrony alerts
            update_callback: Callback for phase state updates
        """
        self.monitor = psi_sync_monitor
        self.synchrony_threshold = synchrony_threshold
        self.update_callback = update_callback
        
        # Mappings between concepts and phase state
        self.concept_to_phase: Dict[str, ConceptPhaseMapping] = {}
        self.phase_to_concept: Dict[int, str] = {}
        
        # Lyapunov functions by concept
        self.concept_lyapunov: Dict[str, List[LyapunovFunction]] = {}
        
        # Current phase state
        self.current_state: Optional[PsiPhaseState] = None
        self.current_metrics: Optional[PsiSyncMetrics] = None
        
        # History for trend analysis
        self.phase_history: Dict[str, List[float]] = {}  # concept_id -> [theta values]
        self.lyapunov_history: Dict[str, List[float]] = {}  # lyapunov_id -> [values]
        
        logger.info("PsiConceptBridge initialized")
        
    def register_concept(
        self,
        concept_id: str,
        phase_index: int,
        psi_mode_indices: List[int],
        psi_mode_weights: Optional[List[float]] = None,
        lyapunov_functions: Optional[List[str]] = None
    ) -> ConceptPhaseMapping:
        """
        Register a concept with the bridge.
        
        This maps a concept in the ELFIN LocalConceptNetwork to
        a phase oscillator and ψ-modes in the phase-space.
        
        Args:
            concept_id: Identifier for the concept
            phase_index: Index in the phase oscillator array
            psi_mode_indices: Indices of ψ-modes that this concept participates in
            psi_mode_weights: Weights for each ψ-mode
            lyapunov_functions: Lyapunov functions that involve this concept
            
        Returns:
            The mapping that was created
        """
        # Create the mapping
        mapping = ConceptPhaseMapping(
            concept_id=concept_id,
            phase_index=phase_index,
            psi_mode_indices=psi_mode_indices,
            psi_mode_weights=psi_mode_weights or [1.0] * len(psi_mode_indices),
            lyapunov_functions=lyapunov_functions or []
        )
        
        # Register mappings
        self.concept_to_phase[concept_id] = mapping
        self.phase_to_concept[phase_index] = concept_id
        
        # Initialize history
        self.phase_history[concept_id] = []
        
        logger.info(f"Registered concept {concept_id} with phase index {phase_index}")
        return mapping
        
    def register_lyapunov_function(
        self,
        lyapunov_fn: LyapunovFunction
    ) -> None:
        """
        Register a Lyapunov function with the bridge.
        
        Args:
            lyapunov_fn: Lyapunov function to register
        """
        # Register with each involved concept
        for concept_id in lyapunov_fn.domain_concept_ids:
            if concept_id not in self.concept_lyapunov:
                self.concept_lyapunov[concept_id] = []
            self.concept_lyapunov[concept_id].append(lyapunov_fn)
            
            # Add to mapping
            if concept_id in self.concept_to_phase:
                if lyapunov_fn.name not in self.concept_to_phase[concept_id].lyapunov_functions:
                    self.concept_to_phase[concept_id].lyapunov_functions.append(lyapunov_fn.name)
        
        # Initialize history
        self.lyapunov_history[lyapunov_fn.name] = []
        
        logger.info(f"Registered Lyapunov function {lyapunov_fn.name} with {len(lyapunov_fn.domain_concept_ids)} concepts")
        
    def update_phase_state(
        self,
        state: PsiPhaseState
    ) -> PsiSyncMetrics:
        """
        Update the current phase state and compute metrics.
        
        This is called when the phase-space state changes, e.g.,
        due to simulation or external input. It updates the
        bridge's internal state and computes metrics.
        
        Args:
            state: New phase state
            
        Returns:
            Synchronization metrics
        """
        # Keep track of old state
        old_state = self.current_state
        self.current_state = state
        
        # Evaluate with the monitor
        metrics = self.monitor.evaluate(state)
        self.current_metrics = metrics
        
        # Update history for each concept
        for concept_id, mapping in self.concept_to_phase.items():
            if mapping.phase_index < len(state.theta):
                theta = state.theta[mapping.phase_index]
                self.phase_history[concept_id].append(theta)
                # Keep history limited
                if len(self.phase_history[concept_id]) > 50:
                    self.phase_history[concept_id].pop(0)
        
        # Update Lyapunov values if we have state data
        if old_state is not None:
            self._update_lyapunov_values(old_state, state)
            
        # If sync is below threshold, create warning
        if metrics.synchrony_score < self.synchrony_threshold:
            logger.warning(f"Low synchrony detected: {metrics.synchrony_score:.2f} < {self.synchrony_threshold}")
            
        # If we have a callback, notify it
        if self.update_callback:
            # Create update for all registered concepts
            concept_ids = list(self.concept_to_phase.keys())
            theta_updates = np.zeros(len(concept_ids))
            psi_updates = np.zeros((len(concept_ids), state.psi.shape[1]))
            
            for i, concept_id in enumerate(concept_ids):
                mapping = self.concept_to_phase[concept_id]
                if mapping.phase_index < len(state.theta):
                    theta_updates[i] = state.theta[mapping.phase_index]
                    
                    # Update psi values (weighted sum of modes)
                    for j, (mode_idx, weight) in enumerate(zip(mapping.psi_mode_indices, mapping.psi_mode_weights)):
                        if mode_idx < state.psi.shape[1]:
                            psi_updates[i, j] = state.psi[mapping.phase_index, mode_idx] * weight
            
            update = PhaseStateUpdate(
                concept_ids=concept_ids,
                theta_updates=theta_updates,
                psi_updates=psi_updates,
                sync_metrics=metrics
            )
            
            self.update_callback(update)
            
        return metrics
        
    def _update_lyapunov_values(
        self,
        old_state: PsiPhaseState,
        new_state: PsiPhaseState
    ) -> None:
        """
        Update Lyapunov function values based on state change.
        
        Args:
            old_state: Previous phase state
            new_state: New phase state
        """
        # We need a special representation for the evaluation
        # For now, just use theta and psi as the state vector
        
        for concept_id, lyapunov_fns in self.concept_lyapunov.items():
            if concept_id not in self.concept_to_phase:
                continue
                
            mapping = self.concept_to_phase[concept_id]
            if mapping.phase_index >= len(new_state.theta):
                continue
                
            # Extract state for this concept
            old_theta = old_state.theta[mapping.phase_index]
            new_theta = new_state.theta[mapping.phase_index]
            
            old_psi = old_state.psi[mapping.phase_index]
            new_psi = new_state.psi[mapping.phase_index]
            
            # Combine into state vectors
            old_x = np.concatenate(([old_theta], old_psi.flatten()))
            new_x = np.concatenate(([new_theta], new_psi.flatten()))
            
            # Evaluate Lyapunov functions
            for lyapunov_fn in lyapunov_fns:
                old_v = lyapunov_fn.evaluate(old_x)
                new_v = lyapunov_fn.evaluate(new_x)
                
                # Store value
                self.lyapunov_history[lyapunov_fn.name].append(new_v)
                if len(self.lyapunov_history[lyapunov_fn.name]) > 50:
                    self.lyapunov_history[lyapunov_fn.name].pop(0)
                    
                # Check if Lyapunov function is decreasing
                if new_v > old_v:
                    logger.warning(f"Lyapunov function {lyapunov_fn.name} increasing: {old_v:.4f} -> {new_v:.4f}")
        
    def apply_concept_update(
        self,
        concept_id: str,
        property_updates: Dict[str, Any]
    ) -> bool:
        """
        Apply a concept state update to the phase-space.
        
        This is called when a concept's state changes in the
        ELFIN LocalConceptNetwork, and the change needs to be
        propagated to the phase-space.
        
        Args:
            concept_id: Identifier for the concept
            property_updates: Updates to concept properties
            
        Returns:
            Whether the update was applied successfully
        """
        if concept_id not in self.concept_to_phase:
            logger.warning(f"Concept {concept_id} not registered with bridge")
            return False
            
        if self.current_state is None:
            logger.warning(f"No current phase state for concept {concept_id}")
            return False
            
        mapping = self.concept_to_phase[concept_id]
        
        # Apply updates to phase state
        # This is a placeholder - a real implementation would
        # interpret the property updates and apply them to the
        # phase state appropriately
        
        logger.info(f"Applied update to concept {concept_id}")
        return True
        
    def get_concept_stability_status(
        self,
        concept_id: str
    ) -> Dict[str, Any]:
        """
        Get the stability status for a concept.
        
        This provides information about the stability of a
        concept based on its phase state and Lyapunov functions.
        
        Args:
            concept_id: Identifier for the concept
            
        Returns:
            Dictionary with stability information
        """
        if concept_id not in self.concept_to_phase:
            return {"status": "unknown", "reason": "Concept not registered"}
            
        if self.current_state is None or self.current_metrics is None:
            return {"status": "unknown", "reason": "No current phase state"}
            
        mapping = self.concept_to_phase[concept_id]
        
        # Basic stability based on phase
        sync_status = "stable"
        if self.current_metrics.synchrony_score < self.synchrony_threshold:
            sync_status = "unstable"
            
        # Lyapunov stability
        lyapunov_status = "unknown"
        if concept_id in self.concept_lyapunov and self.concept_lyapunov[concept_id]:
            # Check if all Lyapunov functions are decreasing
            decreasing = True
            lyapunov_values = {}
            
            for lyapunov_fn in self.concept_lyapunov[concept_id]:
                if lyapunov_fn.name in self.lyapunov_history:
                    history = self.lyapunov_history[lyapunov_fn.name]
                    if len(history) >= 2:
                        prev_v = history[-2]
                        curr_v = history[-1]
                        lyapunov_values[lyapunov_fn.name] = {
                            "current": curr_v,
                            "previous": prev_v,
                            "decreasing": curr_v <= prev_v
                        }
                        if curr_v > prev_v:
                            decreasing = False
                            
            if decreasing:
                lyapunov_status = "stable"
            else:
                lyapunov_status = "unstable"
        
        # Phase stability details
        phase_details = {}
        if concept_id in self.phase_history and len(self.phase_history[concept_id]) >= 2:
            history = self.phase_history[concept_id]
            phase_details = {
                "current_phase": history[-1],
                "phase_drift": history[-1] - history[-2],
                "phase_variance": np.var(history[-10:]) if len(history) >= 10 else 0.0
            }
            
        return {
            "concept_id": concept_id,
            "phase_index": mapping.phase_index,
            "sync_status": sync_status,
            "lyapunov_status": lyapunov_status,
            "synchrony_score": self.current_metrics.synchrony_score,
            "attractor_integrity": self.current_metrics.attractor_integrity,
            "phase_details": phase_details,
            "lyapunov_values": lyapunov_values if 'lyapunov_values' in locals() else {}
        }
        
    def verify_transition(
        self,
        from_concept_id: str,
        to_concept_id: str,
        composite_lyapunov: Optional[CompositeLyapunov] = None
    ) -> bool:
        """
        Verify that a transition between concepts is stable.
        
        This uses the Multi-Agent Lyapunov Guards approach to check
        if transitioning from one concept to another is stable.
        
        Args:
            from_concept_id: Source concept ID
            to_concept_id: Target concept ID
            composite_lyapunov: Optional composite Lyapunov function
            
        Returns:
            Whether the transition is stable
        """
        if from_concept_id not in self.concept_to_phase or to_concept_id not in self.concept_to_phase:
            logger.warning(f"Concepts {from_concept_id} or {to_concept_id} not registered")
            return False
            
        if self.current_state is None:
            logger.warning("No current phase state for transition verification")
            return False
            
        # Get state vectors for each concept
        from_mapping = self.concept_to_phase[from_concept_id]
        to_mapping = self.concept_to_phase[to_concept_id]
        
        from_theta = self.current_state.theta[from_mapping.phase_index]
        from_psi = self.current_state.psi[from_mapping.phase_index]
        
        to_theta = self.current_state.theta[to_mapping.phase_index]
        to_psi = self.current_state.psi[to_mapping.phase_index]
        
        # Combine into state vectors
        from_x = np.concatenate(([from_theta], from_psi.flatten()))
        to_x = np.concatenate(([to_theta], to_psi.flatten()))
        
        # If we have a composite Lyapunov function, use it
        if composite_lyapunov is not None:
            # Find the indices of the components
            from_idx = -1
            to_idx = -1
            
            for i, lyapunov_fn in enumerate(composite_lyapunov.components):
                if from_concept_id in lyapunov_fn.domain_concept_ids:
                    from_idx = i
                if to_concept_id in lyapunov_fn.domain_concept_ids:
                    to_idx = i
                    
            if from_idx >= 0 and to_idx >= 0:
                return composite_lyapunov.verify_transition(from_x, from_idx, to_idx)
                
        # Without a composite Lyapunov function, use a simple heuristic
        # based on phase difference
        phase_diff = np.abs(from_theta - to_theta)
        phase_diff = min(phase_diff, 2 * np.pi - phase_diff)  # Shortest path
        
        # If phases are close, transition is stable
        return phase_diff < 0.5
        
    def recommend_coupling_adjustments(self) -> np.ndarray:
        """
        Recommend adjustments to the coupling matrix.
        
        This uses the current phase state and metrics to recommend
        adjustments to the coupling matrix to improve stability.
        
        Returns:
            Coupling adjustment matrix
        """
        if self.current_state is None or self.current_metrics is None:
            logger.warning("No current phase state for coupling adjustments")
            return None
            
        # Get recommendations from the monitor
        action = self.monitor.recommend_action(self.current_metrics, self.current_state)
        
        return action.coupling_adjustments
        
    def create_phase_state_from_concepts(
        self,
        concept_states: Dict[str, Dict[str, Any]]
    ) -> PsiPhaseState:
        """
        Create a phase state from concept states.
        
        This is used to initialize the phase state based on
        concept states from the ELFIN LocalConceptNetwork.
        
        Args:
            concept_states: Dictionary of concept states
            
        Returns:
            Phase state
        """
        # Determine the size of the state
        max_phase_index = max(
            self.concept_to_phase[concept_id].phase_index
            for concept_id in concept_states
            if concept_id in self.concept_to_phase
        )
        
        # Create arrays
        theta = np.zeros(max_phase_index + 1)
        
        # Determine psi dimension
        max_psi_index = 0
        for concept_id in concept_states:
            if concept_id in self.concept_to_phase:
                mapping = self.concept_to_phase[concept_id]
                if mapping.psi_mode_indices:
                    max_psi_index = max(max_psi_index, max(mapping.psi_mode_indices))
                    
        psi = np.zeros((max_phase_index + 1, max_psi_index + 1))
        
        # Fill in values
        for concept_id, state in concept_states.items():
            if concept_id not in self.concept_to_phase:
                continue
                
            mapping = self.concept_to_phase[concept_id]
            
            # Set theta if provided
            if "theta" in state:
                theta[mapping.phase_index] = state["theta"]
                
            # Set psi if provided
            if "psi" in state:
                for i, mode_idx in enumerate(mapping.psi_mode_indices):
                    if i < len(state["psi"]):
                        psi[mapping.phase_index, mode_idx] = state["psi"][i]
        
        # Create coupling matrix (if necessary)
        coupling_matrix = None
        concept_ids = None
        
        # Create phase state
        return PsiPhaseState(
            theta=theta,
            psi=psi,
            coupling_matrix=coupling_matrix,
            concept_ids=concept_ids
        )
