"""
Koopman Spectral Analysis for ELFIN.

This module implements Extended Dynamic Mode Decomposition (EDMD) for
spectral analysis of system dynamics, identifying dominant modes and 
assessing stability.
"""

import numpy as np
from scipy import linalg
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
import logging
from pathlib import Path
import json

from .snapshot_buffer import SnapshotBuffer

logger = logging.getLogger(__name__)


class EDMDResult(NamedTuple):
    """Results of EDMD spectral decomposition."""
    eigenvalues: np.ndarray  # Complex eigenvalues (λ)
    eigenvectors: np.ndarray  # Eigenvectors (Φ)
    modes: np.ndarray  # Dynamic modes
    amplitudes: np.ndarray  # Mode amplitudes
    frequencies: np.ndarray  # Mode frequencies
    damping_ratios: np.ndarray  # Mode damping ratios
    growth_rates: np.ndarray  # Mode growth/decay rates
    koopman_matrix: np.ndarray  # Koopman operator matrix (K)
    error: float  # Reconstruction error


class SpectralAnalyzer:
    """
    Implements Koopman spectral analysis using EDMD.
    
    This analyzer computes spectral decompositions from snapshot data to
    identify dynamical modes and assess system stability.
    """
    
    def __init__(self, snapshot_buffer: Optional[SnapshotBuffer] = None):
        """
        Initialize the spectral analyzer.
        
        Args:
            snapshot_buffer: Buffer containing system state snapshots
        """
        self.snapshot_buffer = snapshot_buffer
        self.last_result: Optional[EDMDResult] = None
        self.dominant_modes: List[int] = []  # Indices of dominant modes
        self.unstable_modes: List[int] = []  # Indices of unstable modes
    
    def edmd_decompose(self, snapshot_buffer: Optional[SnapshotBuffer] = None, 
                       time_shift: int = 1, svd_rank: Optional[int] = None) -> EDMDResult:
        """
        Perform Extended Dynamic Mode Decomposition.
        
        Args:
            snapshot_buffer: Buffer containing system state snapshots (uses internal if None)
            time_shift: Time shift for constructing data matrices
            svd_rank: Truncation rank for SVD (stability/regularization), None for full rank
            
        Returns:
            EDMDResult containing eigenvalues, modes, and other analysis
            
        Raises:
            ValueError: If snapshot buffer is insufficient for analysis
        """
        buffer = snapshot_buffer or self.snapshot_buffer
        if buffer is None:
            raise ValueError("No snapshot buffer provided")
        
        if len(buffer.buffer) < time_shift + 2:
            raise ValueError(f"Need at least {time_shift + 2} snapshots for EDMD analysis with shift {time_shift}")
        
        # Get time-shifted data matrices
        X, Y = buffer.get_time_shifted_matrices(time_shift)
        
        # Compute SVD of X
        U, Sigma, Vh = linalg.svd(X, full_matrices=False)
        
        # Truncate SVD if requested
        if svd_rank is not None and svd_rank < len(Sigma):
            r = svd_rank
            U = U[:, :r]
            Sigma = Sigma[:r]
            Vh = Vh[:r, :]
        else:
            r = len(Sigma)
        
        # Compute Koopman matrix (standard EDMD formulation)
        # Apply regularization for small singular values
        tol = 1e-10 * Sigma[0]  # Threshold relative to largest singular value
        Sigma_inv = np.diag(np.where(Sigma > tol, 1.0 / Sigma, 0.0))
        
        # Standard EDMD (row-space) operator - no U.T projection
        K = Y @ Vh.T @ Sigma_inv
        
        # Eigendecomposition of Koopman matrix
        eigenvalues, eigenvectors = linalg.eig(K)
        
        # Compute dynamic modes
        modes = Y @ Vh.T @ Sigma_inv @ eigenvectors
        
        # Normalize modes
        for i in range(modes.shape[1]):
            modes[:, i] = modes[:, i] / linalg.norm(modes[:, i])
        
        # Compute mode amplitudes (using first snapshot)
        x0 = X[:, 0]
        b = linalg.lstsq(modes, x0, rcond=None)[0]
        amplitudes = np.abs(b)
        
        # Compute frequencies and damping ratios from eigenvalues
        # Lambda = exp(alpha + i*omega * dt)
        dt = np.mean(np.diff(buffer.timestamps)) if len(buffer.timestamps) > 1 else 1.0
        log_eigs = np.log(eigenvalues) / dt  # Convert to continuous time
        frequencies = np.abs(np.imag(log_eigs)) / (2 * np.pi)  # Hz
        growth_rates = np.real(log_eigs)
        damping_ratios = -np.real(log_eigs) / np.abs(log_eigs)
        
        # Compute reconstruction error with eigenvalue dynamics
        X_reconstructed = np.zeros_like(X)
        for t in range(X.shape[1]):
            # Advance coefficients using eigenvalues
            evolved_coeffs = b * np.power(eigenvalues, t)
            # Reconstruct snapshot at time t
            X_reconstructed[:, t] = modes @ evolved_coeffs
        
        error = linalg.norm(X - X_reconstructed) / linalg.norm(X)
        
        # Create and store result
        result = EDMDResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            modes=modes,
            amplitudes=amplitudes,
            frequencies=frequencies,
            damping_ratios=damping_ratios,
            growth_rates=growth_rates,
            koopman_matrix=K,
            error=error
        )
        
        self.last_result = result
        
        # Update dominant and unstable modes
        self._identify_dominant_modes()
        self._identify_unstable_modes()
        
        return result
    
    def _identify_dominant_modes(self, num_modes: int = 3) -> None:
        """
        Identify the dominant modes based on both amplitude and growth rate.
        
        This uses a weighted scoring system that considers both the initial
        amplitude of the mode and its growth/decay rate.
        
        Args:
            num_modes: Number of dominant modes to identify
        """
        if self.last_result is None:
            return
        
        # Create a score combining amplitude and growth rate
        # Modes with high amplitude AND high growth rate are most important
        amplitudes = self.last_result.amplitudes
        growth_rates = self.last_result.growth_rates
        
        # Normalize both factors to [0, 1] range
        if len(amplitudes) > 0:
            norm_amp = amplitudes / (np.max(amplitudes) if np.max(amplitudes) > 0 else 1.0)
            
            # For growth rates, use exponential scaling to emphasize unstable modes
            # Stable modes (negative growth) get less weight
            scaled_growth = np.exp(np.clip(growth_rates, -2, 2))
            norm_growth = scaled_growth / (np.max(scaled_growth) if np.max(scaled_growth) > 0 else 1.0)
            
            # Combined score (higher is more dominant)
            # Weight amplitude more (0.7) than growth rate (0.3)
            combined_score = 0.7 * norm_amp + 0.3 * norm_growth
            
            # Get indices of highest scoring modes
            idx = np.argsort(combined_score)[::-1]
            self.dominant_modes = idx[:min(num_modes, len(idx))].tolist()
        else:
            self.dominant_modes = []
    
    def _identify_unstable_modes(self, threshold: float = 0.0) -> None:
        """
        Identify unstable modes with positive growth rates.
        
        Args:
            threshold: Growth rate threshold for instability
        """
        if self.last_result is None:
            return
        
        # Find modes with positive growth rates
        self.unstable_modes = np.where(self.last_result.growth_rates > threshold)[0].tolist()
    
    def calculate_stability_index(self) -> float:
        """
        Calculate overall stability index from eigenvalue spectrum.
        
        Returns:
            Stability index between -1 (highly unstable) and 1 (highly stable)
        """
        if self.last_result is None:
            return 0.0
        
        # Use max growth rate as stability indicator
        max_growth = np.max(self.last_result.growth_rates)
        
        # Scale to [-1, 1] range using tanh
        stability_index = -np.tanh(max_growth)
        
        return stability_index
    
    def get_spectral_feedback(self) -> float:
        """
        Generate feedback factor for oscillator coupling based on spectral properties.
        
        Returns:
            Feedback factor (< 1.0 for unstable modes to reduce coupling)
        """
        if self.last_result is None or not self.unstable_modes:
            return 1.0
        
        # Calculate feedback based on instability
        # More unstable → lower feedback to reduce coupling
        stability = self.calculate_stability_index()
        
        # Map from stability [-1, 1] to feedback [0.1, 1.0]
        feedback = 0.55 + 0.45 * (stability + 1) / 2
        
        return feedback
    
    def predict_future_state(self, steps: int = 1) -> np.ndarray:
        """
        Predict future system state using spectral decomposition.
        
        Args:
            steps: Number of time steps into the future to predict
            
        Returns:
            Predicted state vector
        """
        if self.last_result is None or self.snapshot_buffer is None:
            raise ValueError("No analysis results available")
        
        if not self.snapshot_buffer.buffer:
            raise ValueError("Snapshot buffer is empty")
        
        # Get latest state
        latest_state = self.snapshot_buffer.buffer[-1]
        
        # Compute mode coefficients for latest state
        b = linalg.lstsq(self.last_result.modes, latest_state, rcond=None)[0]
        
        # Advance modes by eigenvalues
        advanced_coeffs = b * np.power(self.last_result.eigenvalues, steps)
        
        # Reconstruct predicted state
        predicted_state = self.last_result.modes @ advanced_coeffs
        
        return predicted_state
    
    def export_results(self) -> Dict:
        """
        Export analysis results as a dictionary.
        
        Returns:
            Dictionary containing analysis results
        """
        if self.last_result is None:
            return {'error': 'No analysis results available'}
        
        return {
            'eigenvalues': {
                'real': self.last_result.eigenvalues.real.tolist(),
                'imag': self.last_result.eigenvalues.imag.tolist()
            },
            'amplitudes': self.last_result.amplitudes.tolist(),
            'frequencies': self.last_result.frequencies.tolist(),
            'growth_rates': self.last_result.growth_rates.tolist(),
            'damping_ratios': self.last_result.damping_ratios.tolist(),
            'error': float(self.last_result.error),
            'dominant_modes': self.dominant_modes,
            'unstable_modes': self.unstable_modes,
            'stability_index': self.calculate_stability_index()
        }
    
    def save_results(self, file_path: Path) -> None:
        """
        Save analysis results to file.
        
        Args:
            file_path: Path to save results to
        """
        results = self.export_results()
        
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
