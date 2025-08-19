"""lyapunov_spike_detector.py - Implements spectral stability analysis for inference validation.

This module provides utilities for detecting instability in concept dynamics using
Lyapunov exponent estimation and spectral analysis. It enables:

1. Detection of potential phase desynchronization in concept clusters
2. Estimation of maximum Lyapunov exponents for stability analysis
3. Identification of spectral instabilities in concept dynamics
4. Visualization of stability landscapes and instability signatures

This implements Takata's approach to spectral instability detection, providing
early warning of potential reasoning failures by identifying when concept
dynamics are becoming unstable or chaotic.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass, field
import time
import math
from scipy import stats, signal, linalg
import matplotlib.pyplot as plt
from matplotlib import cm

# Import Koopman components
try:
    # Try absolute import first
    from koopman_estimator import KoopmanEstimator, KoopmanEigenMode
except ImportError:
    # Fallback to relative import
    from .koopman_estimator import KoopmanEstimator, KoopmanEigenMode

# Configure logger
logger = logging.getLogger("lyapunov_spike_detector")

@dataclass
class StabilityAnalysis:
    """
    Results of stability analysis on concept dynamics.
    
    Attributes:
        max_lyapunov: Maximum Lyapunov exponent
        is_stable: Whether the system is considered stable
        instability_risk: Risk score for instability (0.0-1.0)
        confidence: Confidence in the stability assessment (0.0-1.0)
        spectral_gap: Gap between dominant eigenvalues
        largest_eigenvalues: List of largest eigenvalues
        stability_horizon: Time horizon until potential instability
        critical_modes: Indices of modes contributing to instability
    """
    
    max_lyapunov: float = 0.0  # Max Lyapunov exponent
    is_stable: bool = True  # Stability assessment
    instability_risk: float = 0.0  # Risk score
    confidence: float = 1.0  # Confidence in assessment
    spectral_gap: float = 0.0  # Gap between dominant eigenvalues
    largest_eigenvalues: List[complex] = field(default_factory=list)  # Largest eigenvalues
    stability_horizon: float = float('inf')  # Time until instability
    critical_modes: List[int] = field(default_factory=list)  # Modes causing instability
    
    # Detailed stability metrics
    amplitude_variance: float = 0.0  # Variance in mode amplitudes
    resonance_factors: Dict[int, float] = field(default_factory=dict)  # Mode-specific resonance
    stability_time_series: Optional[np.ndarray] = None  # Stability over time
    
    # Secondary Lyapunov spectrum
    lyapunov_spectrum: List[float] = field(default_factory=list)  # All Lyapunov exponents
    
    # Confidence intervals
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    
    def __post_init__(self):
        """Initialize confidence intervals if not provided."""
        if self.confidence_intervals is None:
            self.confidence_intervals = {}
            
        # Calculate stability horizon if max_lyapunov is positive
        if self.max_lyapunov > 0 and self.stability_horizon == float('inf'):
            # Time to double amplitude: ln(2)/lambda
            self.stability_horizon = math.log(2) / max(1e-10, self.max_lyapunov)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of stability analysis results.
        
        Returns:
            Dictionary with key stability metrics
        """
        return {
            "max_lyapunov": self.max_lyapunov,
            "is_stable": self.is_stable,
            "instability_risk": self.instability_risk,
            "confidence": self.confidence,
            "spectral_gap": self.spectral_gap,
            "stability_horizon": self.stability_horizon,
            "critical_modes_count": len(self.critical_modes)
        }
        
    def get_critical_eigenvalues(self) -> List[complex]:
        """
        Get eigenvalues of critical modes causing instability.
        
        Returns:
            List of critical eigenvalues
        """
        if not self.critical_modes or not self.largest_eigenvalues:
            return []
            
        return [
            self.largest_eigenvalues[i]
            for i in self.critical_modes
            if i < len(self.largest_eigenvalues)
        ]

class LyapunovSpikeDetector:
    """
    Detects spectral instabilities in concept dynamics using Lyapunov analysis.
    
    This class implements Takata's approach to stability analysis, providing
    early warning of potential reasoning failures by identifying when concept
    dynamics are becoming unstable or chaotic.
    
    Attributes:
        koopman_estimator: Estimator for Koopman eigenfunctions
        stability_threshold: Maximum Lyapunov exponent threshold for stability
        n_modes: Number of eigenmodes to analyze for stability
        confidence_level: Confidence level for interval estimation
        perturbation_size: Size of perturbation for stability testing
    """
    
    def __init__(
        self,
        koopman_estimator: Optional[KoopmanEstimator] = None,
        stability_threshold: float = 0.01,
        n_modes: int = 5,
        confidence_level: float = 0.95,
        perturbation_size: float = 1e-6
    ):
        """
        Initialize the LyapunovSpikeDetector.
        
        Args:
            koopman_estimator: Koopman eigenfunction estimator
            stability_threshold: Max Lyapunov exponent threshold for stability
            n_modes: Number of eigenmodes to analyze
            confidence_level: Confidence level for interval estimation
            perturbation_size: Size of perturbation for stability testing
        """
        # Create default estimator if not provided
        self.koopman_estimator = koopman_estimator or KoopmanEstimator()
        self.stability_threshold = stability_threshold
        self.n_modes = n_modes
        self.confidence_level = confidence_level
        self.perturbation_size = perturbation_size
        
    def estimate_lyapunov_exponents(
        self,
        trajectory: np.ndarray,
        n_exponents: int = 1,
        method: str = "koopman"
    ) -> np.ndarray:
        """
        Estimate Lyapunov exponents from trajectory data.
        
        Args:
            trajectory: State trajectory with shape (n_samples, n_features)
            n_exponents: Number of Lyapunov exponents to estimate
            method: Method to use ("koopman", "direct", "qr")
            
        Returns:
            Array of Lyapunov exponents
        """
        # Ensure proper shape
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(-1, 1)
            
        if method == "koopman":
            return self._estimate_lyapunov_koopman(trajectory, n_exponents)
        elif method == "direct":
            return self._estimate_lyapunov_direct(trajectory, n_exponents)
        elif method == "qr":
            return self._estimate_lyapunov_qr(trajectory, n_exponents)
        else:
            raise ValueError(f"Unknown method: {method}")
            
    def _estimate_lyapunov_koopman(
        self,
        trajectory: np.ndarray,
        n_exponents: int = 1
    ) -> np.ndarray:
        """
        Estimate Lyapunov exponents using Koopman eigenvalues.
        
        Args:
            trajectory: State trajectory with shape (n_samples, n_features)
            n_exponents: Number of Lyapunov exponents to estimate
            
        Returns:
            Array of Lyapunov exponents
        """
        try:
            # Fit Koopman model to trajectory
            self.koopman_estimator.fit(trajectory)
            
            # Get eigenvalues
            eigenvalues = np.array([
                mode.eigenvalue
                for mode in self.koopman_estimator.eigenmodes
            ])
            
            if len(eigenvalues) == 0:
                # Return zero if no eigenvalues found
                return np.zeros(n_exponents)
                
            # Compute Lyapunov exponents from eigenvalue magnitudes
            # λ = log(|μ|) / dt
            lyapunov = np.log(np.abs(eigenvalues)) / self.koopman_estimator.dt
            
            # Sort by magnitude (descending)
            lyapunov = np.sort(lyapunov)[::-1]
            
            # Return requested number of exponents
            return lyapunov[:n_exponents]
            
        except Exception as e:
            logger.warning(f"Error estimating Lyapunov exponents (Koopman): {e}")
            # Return zero as fallback
            return np.zeros(n_exponents)
            
    def _estimate_lyapunov_direct(
        self,
        trajectory: np.ndarray,
        n_exponents: int = 1
    ) -> np.ndarray:
        """
        Estimate Lyapunov exponents using direct method (perturbation growth).
        
        Args:
            trajectory: State trajectory with shape (n_samples, n_features)
            n_exponents: Number of Lyapunov exponents to estimate
            
        Returns:
            Array of Lyapunov exponents
        """
        try:
            n_samples, n_features = trajectory.shape
            
            # Need at least one trajectory point
            if n_samples < 2:
                return np.zeros(n_exponents)
                
            # Number of exponents limited by dimensionality
            n_exp = min(n_exponents, n_features)
            
            # Initialize perturbation vectors (orthogonal)
            perturbations = np.eye(n_features)[:n_exp] * self.perturbation_size
            
            # Initial reference state
            x_ref = trajectory[0]
            
            # Track growth rates
            log_growth_rates = np.zeros((n_samples - 1, n_exp))
            
            # Evolve perturbations along trajectory
            for t in range(1, n_samples):
                x_next = trajectory[t]
                
                # Evolution map from actual trajectory
                evolution_map = x_next - x_ref
                
                # Apply evolution to perturbations and measure growth
                new_perturbations = np.zeros_like(perturbations)
                
                for i in range(n_exp):
                    # Estimate evolution of perturbation using local linearization
                    # This is a simple approximation of the true evolution
                    pert_evolved = perturbations[i] + evolution_map
                    
                    # Measure growth
                    growth = np.linalg.norm(pert_evolved) / np.linalg.norm(perturbations[i])
                    log_growth_rates[t-1, i] = np.log(growth)
                    
                    # Re-normalize and store
                    new_perturbations[i] = pert_evolved / np.linalg.norm(pert_evolved) * self.perturbation_size
                    
                # Orthogonalize using Gram-Schmidt
                for i in range(n_exp):
                    for j in range(i):
                        proj = np.dot(new_perturbations[i], new_perturbations[j]) / np.dot(new_perturbations[j], new_perturbations[j])
                        new_perturbations[i] = new_perturbations[i] - proj * new_perturbations[j]
                    
                    # Normalize
                    norm = np.linalg.norm(new_perturbations[i])
                    if norm > 0:
                        new_perturbations[i] = new_perturbations[i] / norm * self.perturbation_size
                
                # Update perturbations for next step
                perturbations = new_perturbations
                
                # Update reference state
                x_ref = x_next
                
            # Calculate Lyapunov exponents as average growth rates
            lyapunov = np.mean(log_growth_rates, axis=0)
            
            return lyapunov
            
        except Exception as e:
            logger.warning(f"Error estimating Lyapunov exponents (direct): {e}")
            # Return zero as fallback
            return np.zeros(n_exponents)
            
    def _estimate_lyapunov_qr(
        self,
        trajectory: np.ndarray,
        n_exponents: int = 1
    ) -> np.ndarray:
        """
        Estimate Lyapunov exponents using QR decomposition method.
        
        Args:
            trajectory: State trajectory with shape (n_samples, n_features)
            n_exponents: Number of Lyapunov exponents to estimate
            
        Returns:
            Array of Lyapunov exponents
        """
        try:
            # Fit Koopman model
            self.koopman_estimator.fit(trajectory)
            
            # Get Koopman operator matrix
            K = self.koopman_estimator.koopman_operator
            
            if K is None:
                # Return zero if no operator found
                return np.zeros(n_exponents)
                
            # Get key dimensions
            n_dim = min(K.shape)
            n_exp = min(n_exponents, n_dim)
            
            # Initialize sum of logarithms of diagonal elements
            log_diag_sum = np.zeros(n_dim)
            
            # Number of iterations for convergence
            n_iter = 50
            
            # Initial orthogonal matrix
            Q = np.eye(n_dim)
            
            # QR iterations
            for _ in range(n_iter):
                # Evolve with Koopman operator
                Y = K @ Q
                
                # QR decomposition
                Q, R = np.linalg.qr(Y)
                
                # Extract diagonal of R
                diag = np.diag(R)
                
                # Update sums of logarithms
                log_diag_sum += np.log(np.abs(diag))
                
            # Calculate Lyapunov exponents
            lyapunov = log_diag_sum / (n_iter * self.koopman_estimator.dt)
            
            # Sort by magnitude (descending)
            lyapunov = np.sort(lyapunov)[::-1]
            
            # Return requested number of exponents
            return lyapunov[:n_exp]
            
        except Exception as e:
            logger.warning(f"Error estimating Lyapunov exponents (QR): {e}")
            # Return zero as fallback
            return np.zeros(n_exponents)
            
    def estimate_spectral_gap(
        self,
        eigenvalues: np.ndarray
    ) -> float:
        """
        Estimate spectral gap between dominant eigenvalues.
        
        The spectral gap provides insight into timescale separation and
        potential bifurcations in the dynamics.
        
        Args:
            eigenvalues: Complex eigenvalues
            
        Returns:
            Spectral gap measure
        """
        if len(eigenvalues) < 2:
            return 0.0
            
        # Sort eigenvalues by magnitude (descending)
        mags = np.abs(eigenvalues)
        sorted_idx = np.argsort(mags)[::-1]
        sorted_mags = mags[sorted_idx]
        
        # Compute gap between first and second eigenvalues
        if sorted_mags[0] > 0:
            return (sorted_mags[0] - sorted_mags[1]) / sorted_mags[0]
        else:
            return 0.0
            
    def assess_cluster_stability(
        self,
        cluster_trajectories: List[np.ndarray],
        concept_ids: Optional[List[str]] = None
    ) -> StabilityAnalysis:
        """
        Assess stability of a concept cluster.
        
        Args:
            cluster_trajectories: List of concept trajectories in cluster
            concept_ids: Optional list of concept IDs
            
        Returns:
            StabilityAnalysis with detailed stability metrics
        """
        try:
            # Combine trajectories
            if not cluster_trajectories:
                raise ValueError("No trajectories provided")
                
            # Ensure at least one trajectory
            combined = np.vstack(cluster_trajectories)
            
            # Fit Koopman model
            self.koopman_estimator.fit(combined)
            
            # Get eigenvalues
            eigenvalues = np.array([
                mode.eigenvalue
                for mode in self.koopman_estimator.eigenmodes
            ])
            
            # Compute Lyapunov spectrum
            lyapunov_spectrum = self._estimate_lyapunov_qr(
                combined,
                n_exponents=min(5, combined.shape[1])
            )
            
            # Maximum Lyapunov exponent
            max_lyapunov = lyapunov_spectrum[0] if len(lyapunov_spectrum) > 0 else 0.0
            
            # Stability assessment
            is_stable = max_lyapunov <= self.stability_threshold
            
            # Compute spectral gap
            spectral_gap = self.estimate_spectral_gap(eigenvalues)
            
            # Compute instability risk based on max Lyapunov and spectral gap
            # Higher Lyapunov exponent and smaller spectral gap increase risk
            if max_lyapunov > 0:
                # Scale Lyapunov to [0, 1] range using sigmoid
                lyap_factor = 1.0 / (1.0 + np.exp(-10 * (max_lyapunov - self.stability_threshold)))
                
                # Scale spectral gap (smaller gap → higher risk)
                gap_factor = 1.0 - spectral_gap
                
                # Combine factors with weights
                instability_risk = 0.7 * lyap_factor + 0.3 * gap_factor
            else:
                # Negative Lyapunov indicates stability
                instability_risk = 0.3 * (1.0 - spectral_gap)
                
            # Identify critical modes (modes with positive Lyapunov exponents)
            critical_modes = []
            resonance_factors = {}
            
            for i, mode in enumerate(self.koopman_estimator.eigenmodes):
                # Approximate local Lyapunov exponent for this mode
                mode_lyapunov = np.log(np.abs(mode.eigenvalue)) / self.koopman_estimator.dt
                
                # Check if this mode contributes to instability
                if mode_lyapunov > 0:
                    critical_modes.append(i)
                    
                # Calculate resonance factor for this mode
                # based on how close it is to unit circle
                abs_val = np.abs(mode.eigenvalue)
                resonance_factors[i] = 1.0 - np.abs(1.0 - abs_val)
                
            # Calculate amplitude variance across modes
            amplitudes = np.array([mode.amplitude for mode in self.koopman_estimator.eigenmodes])
            amplitude_variance = np.var(amplitudes) if len(amplitudes) > 0 else 0.0
            
            # Calculate confidence based on estimator confidence
            confidence_values = [mode.confidence for mode in self.koopman_estimator.eigenmodes]
            confidence = np.mean(confidence_values) if confidence_values else 0.5
            
            # Confidence intervals
            confidence_intervals = {
                "max_lyapunov": (
                    max_lyapunov - 0.01,
                    max_lyapunov + 0.01
                ),
                "instability_risk": (
                    max(0.0, instability_risk - 0.1),
                    min(1.0, instability_risk + 0.1)
                )
            }
            
            # Create stability analysis result
            result = StabilityAnalysis(
                max_lyapunov=max_lyapunov,
                is_stable=is_stable,
                instability_risk=instability_risk,
                confidence=confidence,
                spectral_gap=spectral_gap,
                largest_eigenvalues=eigenvalues[:5].tolist() if len(eigenvalues) >= 5 else eigenvalues.tolist(),
                critical_modes=critical_modes,
                amplitude_variance=amplitude_variance,
                resonance_factors=resonance_factors,
                lyapunov_spectrum=lyapunov_spectrum.tolist(),
                confidence_intervals=confidence_intervals
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"Error assessing cluster stability: {e}")
            # Return default stability analysis
            return StabilityAnalysis()
            
    def detect_inference_instability(
        self,
        premise_trajectories: List[np.ndarray],
        conclusion_trajectory: np.ndarray,
        premise_ids: Optional[List[str]] = None,
        conclusion_id: Optional[str] = None
    ) -> Tuple[StabilityAnalysis, StabilityAnalysis, float]:
        """
        Detect instability in inference from premises to conclusion.
        
        Args:
            premise_trajectories: List of premise state trajectories
            conclusion_trajectory: Conclusion state trajectory
            premise_ids: Optional list of premise concept IDs
            conclusion_id: Optional conclusion concept ID
            
        Returns:
            Tuple of (premise stability, combined stability, instability increase)
        """
        # Assess stability of premise cluster
        premise_stability = self.assess_cluster_stability(
            premise_trajectories,
            premise_ids
        )
        
        # Assess stability of combined (premise + conclusion) cluster
        combined_trajectories = premise_trajectories + [conclusion_trajectory]
        combined_ids = None
        
        if premise_ids is not None and conclusion_id is not None:
            combined_ids = premise_ids + [conclusion_id]
            
        combined_stability = self.assess_cluster_stability(
            combined_trajectories,
            combined_ids
        )
        
        # Calculate instability increase
        instability_increase = combined_stability.instability_risk - premise_stability.instability_risk
        
        return premise_stability, combined_stability, instability_increase
        
    def visualize_stability_analysis(
        self,
        analysis: StabilityAnalysis,
        title: str = "Stability Analysis",
        show_plot: bool = True,
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Visualize stability analysis results.
        
        Args:
            analysis: StabilityAnalysis to visualize
            title: Plot title
            show_plot: Whether to display the plot
            save_path: Optional path to save the visualization
            
        Returns:
            Matplotlib figure object if show_plot=False, otherwise None
        """
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        gs = plt.GridSpec(2, 2)
        
        # Plot 1: Lyapunov spectrum
        ax1 = fig.add_subplot(gs[0, 0])
        lyapunov_spectrum = analysis.lyapunov_spectrum
        
        if lyapunov_spectrum:
            x = np.arange(len(lyapunov_spectrum)) + 1
            ax1.bar(x, lyapunov_spectrum, color='purple', alpha=0.7)
            ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            ax1.set_xlabel('Exponent Index')
            ax1.set_ylabel('Lyapunov Exponent')
            ax1.set_title('Lyapunov Spectrum')
            
            # Add unstable/stable annotation
            if analysis.max_lyapunov > 0:
                ax1.text(0.05, 0.95, f"UNSTABLE (λ₁={analysis.max_lyapunov:.4f})",
                         transform=ax1.transAxes, color='red',
                         verticalalignment='top', fontweight='bold')
            else:
                ax1.text(0.05, 0.95, f"STABLE (λ₁={analysis.max_lyapunov:.4f})",
                         transform=ax1.transAxes, color='green',
                         verticalalignment='top', fontweight='bold')
                         
            # Add stability horizon
            if analysis.max_lyapunov > 0:
                ax1.text(0.05, 0.85, f"Stability horizon: {analysis.stability_horizon:.2f} time units",
                         transform=ax1.transAxes, verticalalignment='top')
        else:
            ax1.text(0.5, 0.5, "No Lyapunov spectrum available",
                     ha='center', va='center')
        
        # Plot 2: Eigenvalue spectrum in complex plane
        ax2 = fig.add_subplot(gs[0, 1])
        eigenvalues = analysis.largest_eigenvalues
        
        if eigenvalues:
            # Create unit circle
            theta = np.linspace(0, 2*np.pi, 100)
            ax2.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5)
            
            # Plot eigenvalues
            eig_real = [e.real for e in eigenvalues]
            eig_imag = [e.imag for e in eigenvalues]
            
            # Determine color based on magnitude
            magnitudes = [abs(e) for e in eigenvalues]
            colors = ['red' if m > 1.01 else ('green' if m < 0.99 else 'orange') for m in magnitudes]
            
            ax2.scatter(eig_real, eig_imag, c=colors, s=100, alpha=0.7)
            
            for i, (er, ei, m) in enumerate(zip(eig_real, eig_imag, magnitudes)):
                ax2.annotate(str(i+1), (er, ei), fontsize=8)
            
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlabel('Re(λ)')
            ax2.set_ylabel('Im(λ)')
            ax2.set_title('Eigenvalue Spectrum')
            
            # Set equal aspect ratio to keep circle circular
            ax2.set_aspect('equal')
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='|λ| > 1 (Unstable)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='|λ| ≈ 1 (Neutral)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='|λ| < 1 (Stable)')
            ]
            ax2.legend(handles=legend_elements, loc='lower right')
        else:
            ax2.text(0.5, 0.5, "No eigenvalues available",
                     ha='center', va='center')
        
        # Plot 3: Stability metrics
        ax3 = fig.add_subplot(gs[1, 0])
        metrics = [
            ('Risk', analysis.instability_risk),
            ('Confidence', analysis.confidence),
            ('Spectral Gap', analysis.spectral_gap)
        ]
        
        bars = ax3.bar(
            [m[0] for m in metrics],
            [m[1] for m in metrics],
            color=['red', 'blue', 'purple']
        )
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.2f}',
                ha='center', va='bottom'
            )
            
        ax3.set_ylim(0, 1.1)
        ax3.set_title('Stability Metrics')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Critical modes
        ax4 = fig.add_subplot(gs[1, 1])
        
        if analysis.resonance_factors and analysis.critical_modes:
            # Extract resonance factors for all modes
            modes = sorted(analysis.resonance_factors.keys())
            resonances = [analysis.resonance_factors[m] for m in modes]
            
            # Split into critical and non-critical
            colors = ['red' if m in analysis.critical_modes else 'green' for m in modes]
            
            ax4.bar(modes, resonances, color=colors, alpha=0.7)
            ax4.set_xlabel('Mode Index')
            ax4.set_ylabel('Resonance Factor')
            ax4.set_title('Mode Resonance (Critical Modes in Red)')
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            ax4.text(0.5, 0.5, "No critical modes identified" if not analysis.critical_modes else "No resonance data available",
                     ha='center', va='center')
        
        # Overall title with stability assessment
        stability_status = "STABLE" if analysis.is_stable else "UNSTABLE"
        status_color = "green" if analysis.is_stable else "red"
        
        plt.suptitle(f"{title}\nSystem is {stability_status} (Max Lyapunov: {analysis.max_lyapunov:.4f})",
                   fontsize=16, color=status_color)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
            
        # Show if requested
        if show_plot:
            plt.show()
            return None
            
        return fig
        
    def create_stability_comparison(
        self,
        original_analysis: StabilityAnalysis,
        modified_analysis: StabilityAnalysis,
        title: str = "Stability Comparison",
        show_plot: bool = True,
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Create a visualization comparing two stability analyses.
        
        Args:
            original_analysis: Original stability analysis
            modified_analysis: Modified stability analysis
            title: Plot title
            show_plot: Whether to display the plot
            save_path: Optional path to save the visualization
            
        Returns:
            Matplotlib figure object if show_plot=False, otherwise None
        """
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        gs = plt.GridSpec(2, 2)
        
        # Plot 1: Lyapunov exponent
