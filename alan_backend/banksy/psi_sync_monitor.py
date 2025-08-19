"""psi_sync_monitor.py - Implements real-time ψ-Sync stability monitoring and control.

This module provides a stability monitoring system for the phase-coupled 
oscillator network and Koopman eigenfunction alignment. It enables:

1. Real-time monitoring of phase-spectral coherence
2. Detection of attractor stability and concept drift
3. Adaptive feedback for oscillator coupling adjustment
4. Decision support for inference validation

The PsiSyncMonitor acts as a bridge between the phase oscillator network
and the Koopman spectral analysis, ensuring stable cognitive dynamics.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Literal
from enum import Enum, auto
from dataclasses import dataclass, field
import time
import math

# Configure logger
logger = logging.getLogger("psi_sync_monitor")

@dataclass
class PsiPhaseState:
    """
    Represents the combined state of phase oscillators and Koopman eigenfunctions.
    
    This dataclass encapsulates the dual view of concept state: oscillator phases (θ)
    for memory/context, and Koopman eigenfunctions (ψ) for semantic spectral state.
    
    Attributes:
        theta: Phase values for all oscillators [θ_1, θ_2, ..., θ_N]
        psi: Koopman eigenfunction values for all concepts [ψ_1, ψ_2, ..., ψ_N]
        coupling_matrix: Optional coupling strengths between oscillators (K_ij)
        concept_ids: Optional identifiers for concepts corresponding to oscillators
    """
    
    theta: np.ndarray  # (N,) phases in range [0, 2π)
    psi: np.ndarray  # (N, K) Koopman modes or (N,) for single eigenfunction
    coupling_matrix: Optional[np.ndarray] = None  # (N, N) coupling strengths
    concept_ids: Optional[List[str]] = None  # Concept identifiers
    
    def __post_init__(self):
        """Validate and normalize inputs."""
        # Ensure theta is in range [0, 2π)
        self.theta = self.theta % (2 * np.pi)
        
        # Ensure psi is 2D even for single eigenfunction
        if self.psi.ndim == 1:
            self.psi = self.psi.reshape(-1, 1)
            
        # Validate shapes
        assert len(self.theta) == self.psi.shape[0], "Theta and psi must have same number of concepts"
        
        if self.coupling_matrix is not None:
            assert self.coupling_matrix.shape == (len(self.theta), len(self.theta)), "Coupling matrix shape mismatch"
            
        if self.concept_ids is not None:
            assert len(self.concept_ids) == len(self.theta), "Concept IDs count must match oscillator count"

class SyncState(Enum):
    """
    Enumeration of synchronization stability states.
    
    These states represent the stability status of the oscillator-eigenfunction
    system and guide decision-making in the orchestrator.
    """
    
    STABLE = auto()  # Green zone: high synchrony, reliable inference
    DRIFT = auto()   # Yellow zone: moderate synchrony, speculative inference
    BREAK = auto()   # Red zone: low synchrony, unreliable inference
    UNKNOWN = auto() # Initial or unspecified state

@dataclass
class PsiSyncMetrics:
    """
    Metrics about the phase-eigenfunction synchronization state.
    
    This dataclass contains measurements of various aspects of the system's
    synchronization quality, used for stability assessment and control.
    
    Attributes:
        synchrony_score: Weighted cosine similarity among ψ-aligned concepts
        attractor_integrity: Measure of cluster coherence in ψ-eigenspace
        residual_energy: Deviation from prior Koopman modes
        lyapunov_delta: Rate of change of Lyapunov energy function
        active_mode_indices: Indices of active eigenfunction modes
        active_mode_amplitudes: Amplitudes of active eigenfunction modes
        sync_state: Overall stability assessment
    """
    
    synchrony_score: float = 0.0  # Overall phase synchronization (0-1)
    attractor_integrity: float = 0.0  # Cluster coherence quality (0-1)
    residual_energy: float = 0.0  # Orthogonal projection energy (≥0)
    lyapunov_delta: float = 0.0  # Change in Lyapunov energy function
    active_mode_indices: List[int] = field(default_factory=list)  # Indices of active modes
    active_mode_amplitudes: List[float] = field(default_factory=list)  # Amplitudes of active modes
    sync_state: SyncState = SyncState.UNKNOWN  # Overall stability assessment
    
    def __post_init__(self):
        """Ensure valid metric values."""
        self.synchrony_score = max(0.0, min(1.0, self.synchrony_score))
        self.attractor_integrity = max(0.0, min(1.0, self.attractor_integrity))
        
    def is_stable(self) -> bool:
        """Check if system is in stable sync state."""
        return self.sync_state == SyncState.STABLE
        
    def requires_confirmation(self) -> bool:
        """Check if system needs confirmation due to drift."""
        return self.sync_state == SyncState.DRIFT
        
    def is_unreliable(self) -> bool:
        """Check if system is in unreliable state."""
        return self.sync_state == SyncState.BREAK

@dataclass
class SyncAction:
    """
    Recommended action based on synchronization state.
    
    This dataclass represents recommended actions to take based on the
    system's synchronization state, especially for oscillator coupling
    adjustment.
    
    Attributes:
        coupling_adjustments: Matrix of coupling strength adjustments
        confidence: Confidence in the system's current state (0-1)
        recommendation: Textual recommendation for orchestrator
        requires_user_confirmation: Whether user confirmation is needed
    """
    
    coupling_adjustments: Optional[np.ndarray] = None  # Adjustments to K_ij
    confidence: float = 1.0  # Confidence in current state
    recommendation: str = ""  # Recommendation for orchestrator
    requires_user_confirmation: bool = False  # Whether user confirmation needed

class PsiSyncMonitor:
    """
    Monitors and controls synchronization between phase oscillators and eigenfunctions.
    
    This class bridges the oscillator network (θ) and Koopman eigenfunction (ψ)
    systems, providing real-time stability monitoring and control. It assesses
    the quality of synchronization, identifies attractors, and recommends
    adjustments to maintain cognitive stability.
    
    Attributes:
        stable_threshold: Synchrony threshold for stable state
        drift_threshold: Synchrony threshold for drift state
        residual_threshold: Maximum allowable residual energy
        integrity_threshold: Attractor integrity threshold for stability
        mode_dominance_ratio: Required ratio between top modes
        coupling_learning_rate: Rate for coupling adjustments
        previous_state: Previous PsiPhaseState for tracking changes
    """
    
    def __init__(
        self, 
        stable_threshold: float = 0.9,
        drift_threshold: float = 0.6,
        residual_threshold: float = 0.3,
        integrity_threshold: float = 0.85,
        mode_dominance_ratio: float = 2.0,
        coupling_learning_rate: float = 0.05
    ):
        """
        Initialize the PsiSyncMonitor.
        
        Args:
            stable_threshold: Synchrony threshold for stable state (0-1)
            drift_threshold: Synchrony threshold for drift state (0-1)
            residual_threshold: Maximum residual energy for stability
            integrity_threshold: Attractor integrity threshold (0-1)
            mode_dominance_ratio: Required ratio between top modes
            coupling_learning_rate: Learning rate for coupling adjustments
        """
        self.stable_threshold = stable_threshold
        self.drift_threshold = drift_threshold
        self.residual_threshold = residual_threshold
        self.integrity_threshold = integrity_threshold
        self.mode_dominance_ratio = mode_dominance_ratio
        self.coupling_learning_rate = coupling_learning_rate
        
        # Previous state for tracking changes
        self.previous_state: Optional[PsiPhaseState] = None
        
        # History for Lyapunov trend analysis
        self.lyapunov_history: List[float] = []
        
        # Current metrics
        self.current_metrics: Optional[PsiSyncMetrics] = None
        
        logger.info(
            f"PsiSyncMonitor initialized with thresholds: stable={stable_threshold}, "
            f"drift={drift_threshold}, residual={residual_threshold}, "
            f"integrity={integrity_threshold}"
        )
        
    def evaluate(self, state: PsiPhaseState) -> PsiSyncMetrics:
        """
        Evaluate the current phase-eigenfunction state.
        
        This method computes key synchronization metrics:
        - synchrony_score: How well oscillators align in phase
        - attractor_integrity: How coherent the clusters are in eigenspace
        - residual_energy: How much deviation from earlier Koopman modes
        - lyapunov_delta: Trend in system energy
        
        Args:
            state: Current PsiPhaseState to evaluate
            
        Returns:
            PsiSyncMetrics with computed metrics
        """
        # 1. Calculate phase synchrony score
        synchrony_score = self._compute_synchrony_score(state.theta)
        
        # 2. Calculate attractor integrity
        attractor_integrity = self._compute_attractor_integrity(state)
        
        # 3. Calculate residual energy (if previous state available)
        residual_energy = 0.0
        lyapunov_delta = 0.0
        
        if self.previous_state is not None:
            residual_energy = self._compute_residual_energy(
                state.psi, self.previous_state.psi
            )
            lyapunov_delta = self._compute_lyapunov_delta(state)
        
        # 4. Identify active modes
        active_mode_indices, active_mode_amplitudes = self._identify_active_modes(state.psi)
        
        # 5. Determine overall sync state
        sync_state = self._determine_sync_state(
            synchrony_score, attractor_integrity, residual_energy
        )
        
        # Create metrics object
        metrics = PsiSyncMetrics(
            synchrony_score=synchrony_score,
            attractor_integrity=attractor_integrity,
            residual_energy=residual_energy,
            lyapunov_delta=lyapunov_delta,
            active_mode_indices=active_mode_indices,
            active_mode_amplitudes=active_mode_amplitudes,
            sync_state=sync_state
        )
        
        # Update state
        self.current_metrics = metrics
        self.previous_state = state
        
        # Update Lyapunov history for trend analysis
        current_lyapunov = self._compute_lyapunov_energy(state)
        self.lyapunov_history.append(current_lyapunov)
        if len(self.lyapunov_history) > 20:  # Keep recent history
            self.lyapunov_history.pop(0)
            
        logger.debug(
            f"PsiSync evaluation: synchrony={synchrony_score:.2f}, "
            f"integrity={attractor_integrity:.2f}, residual={residual_energy:.2f}, "
            f"state={sync_state.name}"
        )
            
        return metrics
    
    def recommend_action(self, metrics: PsiSyncMetrics, state: PsiPhaseState) -> SyncAction:
        """
        Recommend actions based on synchronization metrics.
        
        Args:
            metrics: Current synchronization metrics
            state: Current PsiPhaseState
            
        Returns:
            SyncAction with recommended actions
        """
        # Initialize recommendation
        recommendation = ""
        requires_confirmation = False
        
        # No coupling matrix, no adjustments
        if state.coupling_matrix is None:
            coupling_adjustments = None
        else:
            # Initialize coupling adjustments
            coupling_adjustments = np.zeros_like(state.coupling_matrix)
            
            # Only compute adjustments if we have a coupling matrix
            coupling_adjustments = self._compute_coupling_adjustments(state, metrics)
        
        # Determine recommendation based on sync state
        if metrics.sync_state == SyncState.STABLE:
            confidence = min(1.0, metrics.synchrony_score * metrics.attractor_integrity)
            recommendation = "System is stable, proceed with inference"
            requires_confirmation = False
            
        elif metrics.sync_state == SyncState.DRIFT:
            confidence = 0.5 * (metrics.synchrony_score + metrics.attractor_integrity)
            recommendation = "Moderate drift detected, confirm before proceeding"
            requires_confirmation = True
            
        else:  # BREAK or UNKNOWN
            confidence = max(0.0, (metrics.synchrony_score - 0.2) * 2)  # Scale to 0-1
            recommendation = "Significant instability detected, request clarification"
            requires_confirmation = True
        
        return SyncAction(
            coupling_adjustments=coupling_adjustments,
            confidence=confidence,
            recommendation=recommendation,
            requires_user_confirmation=requires_confirmation
        )
        
    def _compute_synchrony_score(self, phases: np.ndarray) -> float:
        """
        Compute the overall phase synchrony score using Kuramoto order parameter.
        
        Args:
            phases: Array of oscillator phases
            
        Returns:
            Synchrony score in [0,1] where 1 means perfect synchrony
        """
        # Compute Kuramoto order parameter r
        complex_sum = np.sum(np.exp(1j * phases))
        r = np.abs(complex_sum) / len(phases)
        return float(r)
        
    def _compute_attractor_integrity(self, state: PsiPhaseState) -> float:
        """
        Compute the integrity of attractors in eigenspace.
        
        This measures how well-formed and separated the clusters are
        in both phase space and eigenfunction space.
        
        Args:
            state: Current PsiPhaseState
            
        Returns:
            Integrity score in [0,1]
        """
        # Simplified approach: use phase differences and eigenfunction alignment
        n_concepts = len(state.theta)
        
        if n_concepts < 2:
            return 1.0  # Single concept is always "coherent"
            
        # Compute phase differences matrix
        theta_diff = np.abs(state.theta[:, np.newaxis] - state.theta[np.newaxis, :])
        # Wrap to [0, π]
        theta_diff = np.minimum(theta_diff, 2*np.pi - theta_diff)
        
        # Compute eigenfunction similarity matrix (using real part for simplicity)
        psi_sim = np.zeros((n_concepts, n_concepts))
        for i in range(n_concepts):
            for j in range(n_concepts):
                # Cosine similarity between eigenfunction vectors
                psi_i = state.psi[i].real
                psi_j = state.psi[j].real
                
                # Avoid division by zero
                norm_i = np.linalg.norm(psi_i)
                norm_j = np.linalg.norm(psi_j)
                
                if norm_i > 0 and norm_j > 0:
                    psi_sim[i, j] = np.dot(psi_i, psi_j) / (norm_i * norm_j)
                else:
                    psi_sim[i, j] = 0.0
        
        # Identify potential clusters based on phase proximity
        threshold_rad = 0.2 * np.pi  # ~36 degrees threshold
        potential_clusters = []
        
        visited = set()
        for i in range(n_concepts):
            if i in visited:
                continue
                
            # Find concepts phase-locked with i
            cluster = [i]
            visited.add(i)
            
            for j in range(n_concepts):
                if j != i and j not in visited and theta_diff[i, j] < threshold_rad:
                    cluster.append(j)
                    visited.add(j)
                    
            if len(cluster) > 1:  # Only consider multi-concept clusters
                potential_clusters.append(cluster)
        
        # Evaluate cluster quality
        cluster_scores = []
        
        for cluster in potential_clusters:
            # Average phase coherence within cluster
            phase_coherence = 0.0
            count = 0
            
            for i in cluster:
                for j in cluster:
                    if i != j:
                        # Convert phase diff to coherence (1 = identical, 0 = opposite)
                        coherence = 1.0 - theta_diff[i, j] / np.pi
                        phase_coherence += coherence
                        count += 1
                        
            if count > 0:
                phase_coherence /= count
                
            # Average eigenfunction alignment within cluster
            psi_alignment = 0.0
            count = 0
            
            for i in cluster:
                for j in cluster:
                    if i != j:
                        psi_alignment += max(0.0, psi_sim[i, j])  # Only positive correlation
                        count += 1
                        
            if count > 0:
                psi_alignment /= count
                
            # Combined score with emphasis on both phase and ψ alignment
            cluster_score = 0.5 * phase_coherence + 0.5 * psi_alignment
            cluster_scores.append(cluster_score)
        
        # Overall integrity is the weighted average of cluster scores
        if cluster_scores:
            # Weight by cluster size (larger clusters matter more)
            weighted_sum = sum(score * len(cluster) for score, cluster in zip(cluster_scores, potential_clusters))
            total_concepts_in_clusters = sum(len(cluster) for cluster in potential_clusters)
            
            if total_concepts_in_clusters > 0:
                integrity = weighted_sum / total_concepts_in_clusters
            else:
                integrity = 0.0
        else:
            # No clusters found - low integrity
            integrity = 0.3  # Some baseline integrity
            
        return integrity
        
    def _compute_residual_energy(self, current_psi: np.ndarray, previous_psi: np.ndarray) -> float:
        """
        Compute residual energy from changes in eigenfunction values.
        
        This measures how much the current psi values deviate from
        previous values in directions orthogonal to the previous subspace.
        
        Args:
            current_psi: Current eigenfunction values
            previous_psi: Previous eigenfunction values
            
        Returns:
            Residual energy (≥0, lower is better)
        """
        # Ensure we're comparing same shape tensors
        if current_psi.shape != previous_psi.shape:
            # Handle shape mismatch (e.g., different number of modes)
            # For simplicity, just use a high residual in this case
            return 1.0
            
        # Compute difference vector
        delta_psi = current_psi - previous_psi
        
        # Normalize by Frobenius norm of previous_psi
        previous_norm = np.linalg.norm(previous_psi)
        if previous_norm > 0:
            normalized_residual = np.linalg.norm(delta_psi) / previous_norm
        else:
            normalized_residual = np.linalg.norm(delta_psi)
            
        return float(normalized_residual)
        
    def _compute_lyapunov_energy(self, state: PsiPhaseState) -> float:
        """
        Compute the Lyapunov energy function V(θ).
        
        The Lyapunov function V(θ) = -½ ∑_i,j K_ij cos(θ_i - θ_j)
        provides a measure that always decreases as the system synchronizes.
        
        Args:
            state: Current PsiPhaseState
            
        Returns:
            Current Lyapunov energy value
        """
        if state.coupling_matrix is None:
            # Without coupling info, use a proxy based on synchrony
            phases = state.theta
            complex_sum = np.sum(np.exp(1j * phases))
            r = np.abs(complex_sum) / len(phases)
            
            # V is inversely related to order parameter r
            return -np.log(r + 1e-10)  # Avoid log(0)
            
        # With coupling matrix, compute V directly
        energy = 0.0
        n = len(state.theta)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    theta_diff = state.theta[i] - state.theta[j]
                    energy -= 0.5 * state.coupling_matrix[i, j] * np.cos(theta_diff)
                    
        return energy
        
    def _compute_lyapunov_delta(self, state: PsiPhaseState) -> float:
        """
        Compute the rate of change of Lyapunov energy.
        
        Args:
            state: Current PsiPhaseState
            
        Returns:
            Change in Lyapunov energy (negative means decreasing energy/increasing stability)
        """
        # Compute current energy
        current_energy = self._compute_lyapunov_energy(state)
        
        # If insufficient history, return zero
        if len(self.lyapunov_history) < 2:
            return 0.0
            
        # Compute average of recent changes
        recent_energies = self.lyapunov_history[-3:]  # Last 3 values
        avg_previous = sum(recent_energies) / len(recent_energies)
        
        return current_energy - avg_previous
        
    def _identify_active_modes(self, psi: np.ndarray) -> Tuple[List[int], List[float]]:
        """
        Identify which eigenfunction modes are currently active.
        
        Args:
            psi: Eigenfunction values
            
        Returns:
            Tuple of (active_mode_indices, active_mode_amplitudes)
        """
        # For multi-mode case (psi has shape n_concepts × n_modes)
        if psi.shape[1] > 1:
            # Compute mode amplitudes (average absolute value across concepts)
            mode_amplitudes = np.mean(np.abs(psi), axis=0)
            
            # Sort modes by amplitude
            sorted_indices = np.argsort(-mode_amplitudes)  # Descending
            
            # Select active modes (amplitude > threshold or top N)
            active_indices = []
            active_amplitudes = []
            
            # Use amplitude threshold (relative to max)
            max_amplitude = mode_amplitudes[sorted_indices[0]]
            threshold = max_amplitude * 0.2  # Mode must be at least 20% of max
            
            for idx in sorted_indices:
                if mode_amplitudes[idx] > threshold:
                    active_indices.append(int(idx))
                    active_amplitudes.append(float(mode_amplitudes[idx]))
                    
            return active_indices, active_amplitudes
            
        # For single mode case
        else:
            # One mode, always active
            return [0], [float(np.mean(np.abs(psi)))]
        
    def _determine_sync_state(
        self, 
        synchrony_score: float, 
        attractor_integrity: float, 
        residual_energy: float
    ) -> SyncState:
        """
        Determine overall synchronization state based on metrics.
        
        Args:
            synchrony_score: Overall phase synchronization score
            attractor_integrity: Cluster coherence quality
            residual_energy: Orthogonal projection energy
            
        Returns:
            SyncState enum value
        """
        # First check residual energy - high values override other metrics
        if residual_energy > self.residual_threshold:
            return SyncState.BREAK
            
        # Next check synchrony with integrity requirement for stable state
        if (synchrony_score >= self.stable_threshold and 
            attractor_integrity >= self.integrity_threshold):
            return SyncState.STABLE
            
        # Check for drift state
        if synchrony_score >= self.drift_threshold:
            return SyncState.DRIFT
            
        # Otherwise, it's a break state
        return SyncState.BREAK
        
    def _compute_coupling_adjustments(
        self, 
        state: PsiPhaseState, 
        metrics: PsiSyncMetrics
    ) -> np.ndarray:
        """
        Compute recommended adjustments to coupling matrix based on ψ alignment.
        
        Implements the feedback rule:
        - Reinforce: K_ij(t+1) ← K_ij + η·cos(ψ_i - ψ_j)
        - Attenuate: if |ψ_i - ψ_j| > π/2, reduce K_ij
        
        Args:
            state: Current PsiPhaseState
            metrics: Current metrics
            
        Returns:
            Matrix of coupling strength adjustments
        """
        if state.coupling_matrix is None:
            return None
            
        n_concepts = len(state.theta)
        adjustments = np.zeros_like(state.coupling_matrix)
        
        # Use alignment of real part of psi for simplicity
        psi_real = state.psi.real
        
        for i in range(n_concepts):
            for j in range(n_concepts):
                if i == j:
                    continue  # Skip self-coupling
                    
                # Compute ψ alignment (use first mode if multiple)
                psi_i = psi_real[i, 0] if psi_real.shape[1] > 0 else psi_real[i]
                psi_j = psi_real[j, 0] if psi_real.shape[1] > 0 else psi_real[j]
                
                # Since we're using real values, use dot product for alignment
                # Normalize to get cosine similarity
                norm_i = np.abs(psi_i)
                norm_j = np.abs(psi_j)
                
                if norm_i > 0 and norm_j > 0:
                    # Cosine similarity in [-1, 1]
                    psi_alignment = (psi_i * psi_j) / (norm_i * norm_j)
                else:
                    psi_alignment = 0.0
                
                # Also consider phase alignment
                phase_diff = state.theta[i] - state.theta[j]
                # Wrap to [-π, π]
                phase_diff = ((phase_diff + np.pi) % (2 * np.pi)) - np.pi
                phase_alignment = np.cos(phase_diff)  # In [-1, 1]
                
                # Combined adjustment factor - reward both alignments
                adjustment = self.coupling_learning_rate * (0.7 * psi_alignment + 0.3 * phase_alignment)
                
                # Apply the adjustment
                adjustments[i, j] = adjustment
                
        return adjustments

def get_psi_sync_monitor(
    stable_threshold: float = 0.9,
    drift_threshold: float = 0.6
) -> PsiSyncMonitor:
    """
    Get or create a PsiSyncMonitor instance with specified parameters.
    
    This function serves as a factory/singleton access point for the monitor.
    
    Args:
        stable_threshold: Synchrony threshold for stable state
        drift_threshold: Synchrony threshold for drift state
        
    Returns:
        PsiSyncMonitor instance
    """
    # This could be enhanced to maintain a singleton or pool of monitors
    return PsiSyncMonitor(
        stable_threshold=stable_threshold,
        drift_threshold=drift_threshold
    )
