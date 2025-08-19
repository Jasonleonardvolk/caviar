"""spectral_monitor.py - Implements real-time cognitive coherence tracking through spectral analysis.

This module provides tools for detecting phase desynchronization and monitoring the coherence 
of ALAN's cognitive processes using spectral methods including:
- Laplacian spectral gap analysis for detecting subsystem fragmentation risk
- Eigenvalue tracking for identifying emergent cognitive patterns
- Koopman decomposition for extracting dynamic modes from state trajectories

References:
- Zhou et al. (2025) for Koopman operator approximation methods
- Ben Arous et al. (2025) for phase transition detection via spectral shifts
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
import logging
from dataclasses import dataclass, field
import warnings
from datetime import datetime

try:
    # Try absolute import first
    from models import ConceptTuple
except ImportError:
    # Fallback to relative import
    from .models import ConceptTuple

# Configure logger
logger = logging.getLogger("alan_spectral_monitor")

@dataclass
class SpectralState:
    """Represents the current spectral state of the cognitive system."""
    timestamp: datetime = field(default_factory=datetime.now)
    spectral_gap: float = 0.0  # Gap between first two eigenvalues
    top_eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))  # Top k eigenvalues
    coherence_score: float = 0.0  # Overall spectral coherence measure
    desync_risk: float = 0.0  # Risk of phase desynchronization (0-1)
    dominant_modes: List[Tuple[int, float]] = field(default_factory=list)  # (mode_id, strength)
    phase_transitions: List[Dict[str, Any]] = field(default_factory=list)  # Detected transitions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "spectral_gap": float(self.spectral_gap),
            "top_eigenvalues": self.top_eigenvalues.tolist() if isinstance(self.top_eigenvalues, np.ndarray) else [],
            "coherence_score": float(self.coherence_score),
            "desync_risk": float(self.desync_risk),
            "dominant_modes": self.dominant_modes,
            "phase_transitions": self.phase_transitions
        }


class SpectralGapAnalyzer:
    """
    Analyzes the Laplacian spectral gap to detect risks of subsystem fragmentation.
    
    The spectral gap (difference between first non-zero and second eigenvalues) 
    indicates how well connected the cognitive network is. A large gap means 
    the system is well-integrated, while a small gap signals vulnerability to
    fragmentation - a potential indicator of cognitive subsystems losing sync.
    """
    
    def __init__(
        self, 
        critical_gap_threshold: float = 0.1,
        n_eigenvalues: int = 5,
        alert_callback: Optional[Callable[[float, float], None]] = None
    ):
        """
        Initialize the spectral gap analyzer.
        
        Args:
            critical_gap_threshold: Threshold below which to consider the gap critical
            n_eigenvalues: Number of eigenvalues to compute
            alert_callback: Optional callback function for gap alerts, taking 
                            (current_gap, threshold) as parameters
        """
        self.critical_gap_threshold = critical_gap_threshold
        self.n_eigenvalues = n_eigenvalues
        self.alert_callback = alert_callback
        self.history = []  # Track gap history
        
    def compute_laplacian(
        self, 
        adjacency_matrix: np.ndarray,
        normalized: bool = True
    ) -> np.ndarray:
        """
        Compute the graph Laplacian from an adjacency matrix.
        
        Args:
            adjacency_matrix: Square matrix with edge weights between nodes
            normalized: Whether to compute the normalized Laplacian
            
        Returns:
            Laplacian matrix as numpy array
        """
        # Convert to sparse if not already
        if not sp.issparse(adjacency_matrix):
            adjacency_matrix = sp.csr_matrix(adjacency_matrix)
            
        # Compute degree matrix
        degrees = adjacency_matrix.sum(axis=1).A1
        D = sp.diags(degrees)
        
        # Compute Laplacian
        L = D - adjacency_matrix
        
        # Normalize if requested
        if normalized:
            # Handle zero degrees (isolated nodes)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                D_sqrt_inv = sp.diags(1.0 / np.sqrt(np.maximum(degrees, 1e-10)))
                L = D_sqrt_inv @ L @ D_sqrt_inv
                
        return L
    
    def compute_spectral_gap(
        self, 
        laplacian: Union[np.ndarray, sp.spmatrix]
    ) -> Tuple[float, np.ndarray]:
        """
        Compute the spectral gap from a Laplacian matrix.
        
        Args:
            laplacian: Graph Laplacian matrix
            
        Returns:
            Tuple of (spectral_gap, eigenvalues)
        """
        if sp.issparse(laplacian):
            # For sparse matrices, use ARPACK
            eigenvalues = spla.eigsh(
                laplacian, 
                k=min(self.n_eigenvalues, laplacian.shape[0]-1),
                which='SM',
                return_eigenvectors=False
            )
        else:
            # For dense matrices, use standard eigenvalue solver
            eigenvalues = np.sort(np.real(la.eigvals(laplacian)))[:self.n_eigenvalues]
            
        # The first eigenvalue of the Laplacian is approximately 0
        # The spectral gap is the difference between the first non-zero eigenvalue
        # and the next one
        if len(eigenvalues) >= 2:
            # Find first non-zero eigenvalue (with small tolerance)
            non_zero_idx = np.where(np.abs(eigenvalues) > 1e-10)[0]
            if len(non_zero_idx) >= 2:
                spectral_gap = eigenvalues[non_zero_idx[1]] - eigenvalues[non_zero_idx[0]]
            else:
                spectral_gap = 0.0
        else:
            spectral_gap = 0.0
            
        return spectral_gap, eigenvalues
        
    def analyze_connectivity(
        self, 
        adjacency_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze the connectivity of a cognitive network.
        
        Args:
            adjacency_matrix: Matrix of connections between cognitive elements
            
        Returns:
            Dictionary with spectral analysis results
        """
        # Compute Laplacian and spectral gap
        laplacian = self.compute_laplacian(adjacency_matrix)
        spectral_gap, eigenvalues = self.compute_spectral_gap(laplacian)
        
        # Assess fragmentation risk
        fragmentation_risk = max(0.0, 1.0 - (spectral_gap / self.critical_gap_threshold))
        
        # Check if gap is critical
        is_critical = spectral_gap < self.critical_gap_threshold
        
        # Add to history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "spectral_gap": spectral_gap,
            "eigenvalues": eigenvalues.tolist(),
            "fragmentation_risk": fragmentation_risk,
            "is_critical": is_critical
        })
        
        # Trigger callback if necessary
        if is_critical and self.alert_callback:
            self.alert_callback(spectral_gap, self.critical_gap_threshold)
            
        # Log result
        if is_critical:
            logger.warning(f"Critical spectral gap detected: {spectral_gap:.4f} " 
                          f"(threshold: {self.critical_gap_threshold:.4f})")
        else:
            logger.debug(f"Spectral gap: {spectral_gap:.4f} " 
                        f"(threshold: {self.critical_gap_threshold:.4f})")
            
        return {
            "spectral_gap": spectral_gap,
            "eigenvalues": eigenvalues.tolist(),
            "fragmentation_risk": fragmentation_risk,
            "is_critical": is_critical
        }
    
    def build_concept_adjacency(
        self, 
        concepts: List[ConceptTuple],
        similarity_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Build an adjacency matrix from concept relationships.
        
        Args:
            concepts: List of ConceptTuple objects
            similarity_threshold: Minimum similarity to consider an edge
            
        Returns:
            Adjacency matrix for the concept graph
        """
        n = len(concepts)
        adjacency = np.zeros((n, n))
        
        # Compute concept similarities
        for i in range(n):
            for j in range(i+1, n):
                # Calculate cosine similarity between embeddings
                if hasattr(concepts[i], 'embedding') and hasattr(concepts[j], 'embedding'):
                    sim = np.dot(concepts[i].embedding, concepts[j].embedding) / (
                        np.linalg.norm(concepts[i].embedding) * np.linalg.norm(concepts[j].embedding)
                    )
                    if sim > similarity_threshold:
                        adjacency[i, j] = sim
                        adjacency[j, i] = sim
                        
        return adjacency
        
    def get_fragmentation_trend(self, window: int = 10) -> Dict[str, Any]:
        """
        Analyze the trend in spectral gap over time.
        
        Args:
            window: Number of recent observations to consider
            
        Returns:
            Dictionary with trend information
        """
        if len(self.history) < 2:
            return {"status": "insufficient_data"}
            
        # Get recent gap values
        recent = self.history[-min(window, len(self.history)):]
        gaps = [entry["spectral_gap"] for entry in recent]
        
        # Simple trend calculation
        start, end = gaps[0], gaps[-1]
        change = end - start
        percent_change = 100 * change / start if abs(start) > 1e-10 else float('inf')
        
        # Determine trend direction
        if abs(percent_change) < 10:
            trend = "stable"
        elif percent_change > 0:
            trend = "improving"
        else:
            trend = "deteriorating"
            
        return {
            "status": "analyzed",
            "trend": trend,
            "start_value": start,
            "current_value": end,
            "change": change,
            "percent_change": percent_change,
            "data_points": len(gaps)
        }


class EigenmodeTrendTracker:
    """
    Tracks the evolution of eigenvalues over time to detect phase transitions.
    
    Based on insights from Ben Arous et al. (2025), this class monitors for sudden
    shifts in the eigenspectrum that might indicate transitions in the cognitive state,
    such as the emergence of new concepts or the splitting of existing ones.
    """
    
    def __init__(
        self, 
        n_modes: int = 10,
        transition_threshold: float = 0.3
    ):
        """
        Initialize the eigenmode tracker.
        
        Args:
            n_modes: Number of top eigenmodes to track
            transition_threshold: Threshold for significant eigenvalue change
        """
        self.n_modes = n_modes
        self.transition_threshold = transition_threshold
        self.eigenvalue_history = []
        self.transition_history = []
        
    def record_eigenvalues(
        self, 
        eigenvalues: np.ndarray,
        state_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a new set of eigenvalues and detect transitions.
        
        Args:
            eigenvalues: Array of eigenvalues
            state_metadata: Optional metadata about the current state
        """
        # Ensure we're tracking the top n_modes
        if len(eigenvalues) > self.n_modes:
            eigenvalues = np.sort(eigenvalues)[:self.n_modes]
            
        # Convert to list if needed
        eigenvalues_list = eigenvalues.tolist() if isinstance(eigenvalues, np.ndarray) else eigenvalues
        
        # Record the eigenvalues with timestamp
        record = {
            "timestamp": datetime.now().isoformat(),
            "eigenvalues": eigenvalues_list,
            "metadata": state_metadata or {}
        }
        self.eigenvalue_history.append(record)
        
        # Detect transitions if we have history
        if len(self.eigenvalue_history) >= 2:
            self._detect_transitions(record)
    
    def _detect_transitions(self, current_record: Dict[str, Any]) -> None:
        """
        Detect significant transitions in the eigenspectrum.
        
        Args:
            current_record: Current eigenvalue record
        """
        # Get previous record
        prev_record = self.eigenvalue_history[-2]
        
        # Get eigenvalues
        current_evals = np.array(current_record["eigenvalues"])
        prev_evals = np.array(prev_record["eigenvalues"])
        
        # Ensure matching lengths
        min_len = min(len(current_evals), len(prev_evals))
        current_evals = current_evals[:min_len]
        prev_evals = prev_evals[:min_len]
        
        # Calculate relative changes
        if np.all(np.abs(prev_evals) < 1e-10):
            # Previous values are very close to zero, can't compute relative change
            rel_changes = np.zeros_like(current_evals)
        else:
            rel_changes = np.abs((current_evals - prev_evals) / np.maximum(np.abs(prev_evals), 1e-10))
            
        # Detect significant changes
        significant_modes = np.where(rel_changes > self.transition_threshold)[0]
        
        if len(significant_modes) > 0:
            # Record the transition
            transition = {
                "timestamp": current_record["timestamp"],
                "modes": significant_modes.tolist(),
                "previous_values": prev_evals[significant_modes].tolist(),
                "current_values": current_evals[significant_modes].tolist(),
                "relative_changes": rel_changes[significant_modes].tolist(),
                "metadata": current_record.get("metadata", {})
            }
            self.transition_history.append(transition)
            
            # Log the transition
            logger.info(f"Eigenmode transition detected in modes {transition['modes']} "
                      f"with relative changes {transition['relative_changes']}")
    
    def get_recent_transitions(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent transitions.
        
        Args:
            limit: Maximum number of transitions to return
            
        Returns:
            List of recent transitions
        """
        return self.transition_history[-limit:]
    
    def get_active_modes(self) -> List[Tuple[int, float]]:
        """
        Get the currently active modes based on recent eigenvalues.
        
        Returns:
            List of (mode_index, eigenvalue) tuples for active modes
        """
        if not self.eigenvalue_history:
            return []
            
        # Get most recent eigenvalues
        eigenvalues = np.array(self.eigenvalue_history[-1]["eigenvalues"])
        
        # Find active modes (significant magnitudes)
        active_indices = np.where(np.abs(eigenvalues) > 1e-6)[0]
        
        # Sort by absolute magnitude (descending)
        active_indices = active_indices[np.argsort(-np.abs(eigenvalues[active_indices]))]
        
        # Return as (index, value) pairs
        return [(int(i), float(eigenvalues[i])) for i in active_indices]


class KoopmanDecomposer:
    """
    Extracts dynamic modes from system trajectories using Koopman analysis.
    
    Based on Zhou et al. (2025), this implements Koopman operator approximation
    to identify the fundamental modes governing the system's dynamics, enabling
    the detection of periodic or quasi-periodic cycles, slowly decaying trends,
    or chaotic dynamics.
    """
    
    def __init__(
        self, 
        state_dim: int,
        n_modes: int = 5,
        dt: float = 1.0,
        yosida_parameter: float = 0.1
    ):
        """
        Initialize the Koopman decomposer.
        
        Args:
            state_dim: Dimension of the state vector
            n_modes: Number of Koopman modes to extract
            dt: Time step between observations
            yosida_parameter: Parameter for Yosida approximation (Zhou et al.)
        """
        self.state_dim = state_dim
        self.n_modes = n_modes
        self.dt = dt
        self.yosida_parameter = yosida_parameter
        self.trajectory_history = []
        self.koopman_matrix = None
        self.eigenvalues = None
        self.modes = None
        
    def record_state(self, state_vector: np.ndarray) -> None:
        """
        Record a new state observation.
        
        Args:
            state_vector: Current state of the system
        """
        # Ensure state has correct shape
        state_vector = np.asarray(state_vector).flatten()
        if state_vector.shape[0] != self.state_dim:
            raise ValueError(f"State vector must have dimension {self.state_dim}, "
                           f"got {state_vector.shape[0]}")
                           
        # Record the state with timestamp
        self.trajectory_history.append({
            "timestamp": datetime.now().isoformat(),
            "state": state_vector.tolist()
        })
        
    def compute_koopman_approximation(self) -> Dict[str, Any]:
        """
        Compute Koopman operator approximation from trajectory data.
        
        Based on Zhou et al. (2025) Yosida approximation method.
        
        Returns:
            Dictionary with Koopman analysis results
        """
        if len(self.trajectory_history) < 2:
            return {"status": "insufficient_data"}
            
        # Extract trajectory data
        states = np.array([np.array(record["state"]) for record in self.trajectory_history])
        
        # Compute state differences and mid-points
        x_t = states[:-1]  # States at time t
        x_t_plus = states[1:]  # States at time t+1
        
        # Apply Yosida approximation for Koopman generator
        # K ≈ (I - λG)^(-1) where G is the generator, λ is Yosida parameter
        # G ≈ (1/λ)(K - I)
        # Following Zhou et al. approach
        
        # Compute K using least squares regression
        # K maps x_t to x_t_plus
        K = np.linalg.lstsq(x_t, x_t_plus, rcond=None)[0].T
        
        # Apply Yosida approximation to get generator
        I = np.eye(K.shape[0])
        G = (1.0 / self.yosida_parameter) * (K - I)
        
        # Compute eigendecomposition of the generator
        evals, evecs = la.eig(G)
        
        # Sort by real part (descending)
        idx = np.argsort(-np.real(evals))
        evals = evals[idx]
        evecs = evecs[:, idx]
        
        # Store results
        self.koopman_matrix = G
        self.eigenvalues = evals[:self.n_modes]
        self.modes = evecs[:, :self.n_modes]
        
        # Create result dictionary
        modes_data = []
        for i in range(min(self.n_modes, len(evals))):
            mode_dict = {
                "index": i,
                "eigenvalue": {
                    "real": float(np.real(evals[i])),
                    "imag": float(np.imag(evals[i]))
                },
                "period": None,
                "decay_rate": float(np.real(evals[i])),
                "oscillation_freq": float(np.imag(evals[i])) if np.imag(evals[i]) != 0 else None,
            }
            
            # Calculate period for oscillatory modes
            if np.imag(evals[i]) != 0:
                mode_dict["period"] = float(2 * np.pi / abs(np.imag(evals[i])))
                
            modes_data.append(mode_dict)
            
        result = {
            "status": "success",
            "n_samples": len(self.trajectory_history),
            "modes": modes_data,
            "spectrum_type": self._classify_spectrum(self.eigenvalues)
        }
        
        return result
        
    def _classify_spectrum(self, eigenvalues: np.ndarray) -> str:
        """
        Classify the Koopman spectrum type.
        
        Args:
            eigenvalues: Koopman eigenvalues
            
        Returns:
            String describing the spectrum type
        """
        # Check for discrete vs. continuous spectrum
        if len(eigenvalues) == 0:
            return "unknown"
            
        # Check for dominant oscillatory modes
        has_oscillatory = np.any(np.abs(np.imag(eigenvalues)) > 1e-6)
        
        # Check for slow decay
        has_slow_decay = np.any((np.real(eigenvalues) > -1e-4) & (np.real(eigenvalues) < 0))
        
        # Check for unstable modes
        has_unstable = np.any(np.real(eigenvalues) > 1e-6)
        
        # Classify spectrum
        if has_unstable:
            return "unstable"
        elif has_oscillatory and has_slow_decay:
            return "damped_oscillatory"
        elif has_oscillatory:
            return "oscillatory"
        elif has_slow_decay:
            return "slow_decay"
        else:
            return "fast_decay"
        
    def get_dominant_modes(self) -> List[Dict[str, Any]]:
        """
        Get the dominant Koopman modes.
        
        Returns:
            List of dominant modes with their properties
        """
        if self.eigenvalues is None or self.modes is None:
            return []
            
        # Create mode data
        modes_data = []
        for i in range(min(self.n_modes, len(self.eigenvalues))):
            eig = self.eigenvalues[i]
            mode = self.modes[:, i]
            
            # Calculate properties
            decay_rate = float(np.real(eig))
            oscillation_freq = float(np.imag(eig)) if np.imag(eig) != 0 else None
            period = float(2 * np.pi / abs(np.imag(eig))) if np.imag(eig) != 0 else None
            
            # Find top components of this mode
            mode_mag = np.abs(mode)
            top_indices = np.argsort(-mode_mag)[:3]  # Top 3 components
            
            # Create mode dictionary
            mode_dict = {
                "index": i,
                "eigenvalue": {
                    "real": float(np.real(eig)),
                    "imag": float(np.imag(eig))
                },
                "decay_rate": decay_rate,
                "oscillation_freq": oscillation_freq,
                "period": period,
                "top_components": [int(idx) for idx in top_indices],
                "stability": "unstable" if decay_rate > 0 else "stable"
            }
            
            modes_data.append(mode_dict)
            
        return modes_data

    def project_state(self, state_vector: np.ndarray) -> Dict[str, float]:
        """
        Project a state vector onto the Koopman modes.
        
        Args:
            state_vector: State to project
            
        Returns:
            Dictionary mapping mode indices to projection magnitudes
        """
        if self.modes is None:
            return {}
            
        # Ensure state has correct shape
        state_vector = np.asarray(state_vector).flatten()
        if state_vector.shape[0] != self.state_dim:
            raise ValueError(f"State vector must have dimension {self.state_dim}, "
                           f"got {state_vector.shape[0]}")
                           
        # Calculate mode projections (simplified)
        projections = {}
        for i in range(self.modes.shape[1]):
            mode = self.modes[:, i]
            # Use absolute inner product as projection magnitude
            proj = np.abs(np.dot(state_vector, mode.conj()))
            projections[i] = float(proj)
            
        return projections

    def predict_future_state(self, state_vector: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Predict a future state using the learned Koopman dynamics.
        
        Args:
            state_vector: Current state
            steps: Number of steps to predict ahead
            
        Returns:
            Predicted future state
        """
        if self.koopman_matrix is None:
            return state_vector
            
        # Ensure state has correct shape
        state_vector = np.asarray(state_vector).flatten()
        if state_vector.shape[0] != self.state_dim:
            raise ValueError(f"State vector must have dimension {self.state_dim}, "
                           f"got {state_vector.shape[0]}")
                           
        # Compute matrix exponential: exp(G*t)
        time = steps * self.dt
        expm_Gt = la.expm(self.koopman_matrix * time)
        
        # Apply to state: exp(G*t) * x
        future_state = expm_Gt @ state_vector
        
        return future_state


class CognitiveSpectralMonitor:
    """
    Main class integrating all spectral monitoring components.
    
    This provides a unified interface for monitoring the spectral properties
    of ALAN's cognitive processes, detecting phase desynchronization risks,
    and tracking the evolution of cognitive modes over time.
    """
    
    def __init__(
        self,
        state_dim: Optional[int] = None,
        n_modes: int = 5,
        log_dir: str = "logs/spectral_monitor"
    ):
        """
        Initialize the cognitive spectral monitor.
        
        Args:
            state_dim: Dimension of the state vector for Koopman analysis
            n_modes: Number of modes to track
            log_dir: Directory for logging spectral data
        """
        self.n_modes = n_modes
        self.log_dir = log_dir
        
        # Initialize components
        self.gap_analyzer = SpectralGapAnalyzer(n_eigenvalues=n_modes)
        self.mode_tracker = EigenmodeTrendTracker(n_modes=n_modes)
        
        # Initialize Koopman analyzer if state dimension is provided
        self.koopman_analyzer = None
        if state_dim is not None:
            self.koopman_analyzer = KoopmanDecomposer(state_dim=state_dim, n_modes=n_modes)
        
        # Track spectral state history
        self.spectral_states = []
        
        logger.info("Cognitive spectral monitor initialized")
        
    def set_state_dimension(self, state_dim: int) -> None:
        """
        Set the state dimension for Koopman analysis.
        
        Args:
            state_dim: Dimension of the state vector
        """
        self.koopman_analyzer = KoopmanDecomposer(state_dim=state_dim, n_modes=self.n_modes)
        logger.info(f"Koopman analyzer initialized with state dimension {state_dim}")
        
    def analyze_connectivity(
        self,
        adjacency_matrix: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze the connectivity structure of a cognitive network.
        
        Args:
            adjacency_matrix: Matrix representing connections between elements
            metadata: Optional metadata about the current context
            
        Returns:
            Dictionary with analysis results
        """
        # Analyze spectral gap
        gap_results = self.gap_analyzer.analyze_connectivity(adjacency_matrix)
        
        # Track eigenvalues
        self.mode_tracker.record_eigenvalues(
            np.array(gap_results["eigenvalues"]),
            metadata
        )
        
        # Get active modes
        active_modes = self.mode_tracker.get_active_modes()
        
        # Get recent transitions
        recent_transitions = self.mode_tracker.get_recent_transitions(limit=3)
        
        # Create spectral state
        state = SpectralState(
            timestamp=datetime.now(),
            spectral_gap=gap_results["spectral_gap"],
            top_eigenvalues=np.array(gap_results["eigenvalues"]),
            coherence_score=1.0 - gap_results["fragmentation_risk"],
            desync_risk=gap_results["fragmentation_risk"],
            dominant_modes=active_modes,
            phase_transitions=recent_transitions
        )
        
        # Save state
        self.spectral_states.append(state)
        
        # Return combined results
        return {
            "spectral_gap": gap_results["spectral_gap"],
            "fragmentation_risk": gap_results["fragmentation_risk"],
            "is_critical": gap_results["is_critical"],
            "active_modes": active_modes,
            "recent_transitions": recent_transitions,
            "coherence_score": state.coherence_score
        }
        
    def record_state_vector(self, state_vector: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Record a state vector for Koopman analysis.
        
        Args:
            state_vector: Current cognitive state vector
            
        Returns:
            Analysis results if Koopman analysis is performed, None otherwise
        """
        if self.koopman_analyzer is None:
            logger.warning("Koopman analyzer not initialized, state vector ignored")
            return None
            
        # Record state
        self.koopman_analyzer.record_state(state_vector)
        
        # Only compute Koopman approximation occasionally (every 10 states)
        result = None
        if len(self.koopman_analyzer.trajectory_history) % 10 == 0:
            result = self.koopman_analyzer.compute_koopman_approximation()
            
            # Update spectral state with Koopman modes if successful
            if result["status"] == "success":
                # Extract dominant modes
                dominant_modes = []
                for mode in result["modes"]:
                    dominant_modes.append((mode["index"], mode["decay_rate"]))
                
                # Create new spectral state
                state = SpectralState(
                    timestamp=datetime.now(),
                    spectral_gap=0.0,  # No direct spectral gap from Koopman
                    top_eigenvalues=np.array([mode["eigenvalue"]["real"] + 1j * mode["eigenvalue"]["imag"] 
                                            for mode in result["modes"]]),
                    coherence_score=0.8 if result["spectrum_type"] in ["slow_decay", "damped_oscillatory"] else 0.5,
                    desync_risk=1.0 if result["spectrum_type"] == "unstable" else 0.2,
                    dominant_modes=dominant_modes,
                    phase_transitions=[]  # No transitions detected here
                )
                
                # Save state
                self.spectral_states.append(state)
                
        return result
        
    def analyze_concept_network(self, concepts: List[ConceptTuple]) -> Dict[str, Any]:
        """
        Analyze the spectral properties of a concept network.
        
        This is a convenience method that builds the adjacency matrix from concepts
        and then performs spectral analysis on it.
        
        Args:
            concepts: List of concepts to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if len(concepts) < 2:
            return {
                "status": "insufficient_concepts",
                "message": f"Need at least 2 concepts, got {len(concepts)}"
            }
            
        # Build adjacency matrix
        adjacency = self.gap_analyzer.build_concept_adjacency(concepts)
        
        # Analyze connectivity
        metadata = {
            "concept_count": len(concepts),
            "context": "concept_network"
        }
        return self.analyze_connectivity(adjacency, metadata)
        
    def get_current_spectral_state(self) -> Dict[str, Any]:
        """
        Get the current spectral state of the system.
        
        Returns:
            Dictionary with current spectral state
        """
        if not self.spectral_states:
            return {
                "status": "no_data",
                "message": "No spectral states recorded yet"
            }
            
        # Get most recent state
        state = self.spectral_states[-1]
        
        # Get spectral gap trend
        gap_trend = self.gap_analyzer.get_fragmentation_trend()
        
        return {
            "status": "success",
            "timestamp": state.timestamp.isoformat(),
            "spectral_gap": state.spectral_gap,
            "coherence_score": state.coherence_score,
            "desync_risk": state.desync_risk,
            "dominant_modes": state.dominant_modes,
            "recent_transitions": state.phase_transitions,
            "gap_trend": gap_trend.get("trend", "unknown") if gap_trend.get("status") == "analyzed" else "unknown"
        }
        
    def predict_state_evolution(
        self, 
        state_vector: np.ndarray, 
        steps: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Predict how a state will evolve over time using Koopman dynamics.
        
        Args:
            state_vector: Current state
            steps: Number of steps to predict ahead
            
        Returns:
            Dictionary with prediction results, or None if Koopman analyzer not initialized
        """
        if self.koopman_analyzer is None or self.koopman_analyzer.koopman_matrix is None:
            return None
            
        # Make predictions for each step
        states = [state_vector]
        for i in range(steps):
            next_state = self.koopman_analyzer.predict_future_state(states[-1])
            states.append(next_state)
            
        # Calculate projections for each state
        projections = []
        for state in states:
            proj = self.koopman_analyzer.project_state(state)
            projections.append(proj)
            
        return {
            "initial_state": state_vector.tolist(),
            "predicted_states": [state.tolist() for state in states[1:]],
            "state_projections": projections,
            "steps": steps,
            "spectrum_type": self.koopman_analyzer._classify_spectrum(self.koopman_analyzer.eigenvalues)
        }
        
    def summarize_spectral_trends(self, window: int = 10) -> Dict[str, Any]:
        """
        Generate a summary of recent spectral trends.
        
        Args:
            window: Number of recent states to consider
            
        Returns:
            Dictionary with trend summary
        """
        if len(self.spectral_states) < 2:
            return {
                "status": "insufficient_data",
                "message": f"Need at least 2 spectral states, got {len(self.spectral_states)}"
            }
            
        # Get recent states
        recent = self.spectral_states[-min(window, len(self.spectral_states)):]
        
        # Calculate average metrics
        avg_gap = sum(state.spectral_gap for state in recent) / len(recent)
        avg_coherence = sum(state.coherence_score for state in recent) / len(recent)
        avg_risk = sum(state.desync_risk for state in recent) / len(recent)
        
        # Calculate trends (simple linear trend)
        gap_trend = (recent[-1].spectral_gap - recent[0].spectral_gap) / max(1e-10, recent[0].spectral_gap)
        coherence_trend = (recent[-1].coherence_score - recent[0].coherence_score) / max(1e-10, recent[0].coherence_score)
        risk_trend = (recent[-1].desync_risk - recent[0].desync_risk) / max(1e-10, recent[0].desync_risk)
        
        # Count transitions
        transitions = sum(len(state.phase_transitions) for state in recent)
        
        # Count active modes
        active_modes = set()
        for state in recent:
            for mode_idx, _ in state.dominant_modes:
                active_modes.add(mode_idx)
                
        return {
            "status": "success",
            "window_size": len(recent),
            "time_span": (recent[-1].timestamp - recent[0].timestamp).total_seconds(),
            "metrics": {
                "average_spectral_gap": avg_gap,
                "average_coherence_score": avg_coherence,
                "average_desync_risk": avg_risk
            },
            "trends": {
                "spectral_gap": gap_trend,
                "coherence_score": coherence_trend,
                "desync_risk": risk_trend
            },
            "transition_count": transitions,
            "active_mode_count": len(active_modes),
            "active_modes": list(active_modes)
        }

# Singleton instance for easy access
_cognitive_spectral_monitor = None

def get_cognitive_spectral_monitor(
    state_dim: Optional[int] = None, 
    n_modes: int = 5
) -> CognitiveSpectralMonitor:
    """
    Get or create the singleton spectral monitor instance.
    
    Args:
        state_dim: Optional dimension of state vector for Koopman analysis
        n_modes: Number of modes to track
        
    Returns:
        CognitiveSpectralMonitor instance
    """
    global _cognitive_spectral_monitor
    if _cognitive_spectral_monitor is None:
        _cognitive_spectral_monitor = CognitiveSpectralMonitor(
            state_dim=state_dim,
            n_modes=n_modes
        )
    elif state_dim is not None and _cognitive_spectral_monitor.koopman_analyzer is None:
        # If state_dim provided and Koopman not yet initialized
        _cognitive_spectral_monitor.set_state_dimension(state_dim)
        
    return _cognitive_spectral_monitor
