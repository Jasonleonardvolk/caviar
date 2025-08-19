"""
Integration between Koopman eigenfunction estimation and ψ-Sync stability monitoring.

This module demonstrates how to:
1. Use the KoopmanEstimator to compute eigenfunctions (ψ)
2. Feed these into the PsiSyncMonitor
3. Evaluate stability of the cognitive state
4. Adjust oscillator coupling based on eigenfunction alignment
5. Make stability-aware inferences

This provides a complete example of the ψ-enriched cognitive architecture.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("psi_koopman_integration")

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our modules
from alan_backend.banksy import (
    PsiSyncMonitor, 
    PsiPhaseState, 
    PsiSyncMetrics, 
    SyncAction,
    SyncState
)

# Import Koopman estimator
from ingest_pdf.koopman_estimator import (
    KoopmanEstimator,
    KoopmanEigenMode,
    BasisFunction
)

class PsiKoopmanIntegrator:
    """
    Integrates Koopman eigenfunction estimation with ψ-Sync stability monitoring.
    
    This class serves as the bridge between the spectral analysis provided by
    KoopmanEstimator and the phase stability monitoring provided by PsiSyncMonitor.
    It handles:
    
    1. Computing Koopman eigenfunctions from time series data
    2. Extracting phase information from oscillators
    3. Feeding both into PsiSyncMonitor
    4. Providing stability-aware concept recommendations
    
    Attributes:
        koopman_estimator: KoopmanEstimator for eigenfunction computation
        sync_monitor: PsiSyncMonitor for phase-eigenfunction sync monitoring
        concept_metadata: Optional metadata for concepts
    """
    
    def __init__(
        self,
        basis_type: str = "fourier",
        n_eigenfunctions: int = 3,
        time_step: float = 1.0,
        stable_threshold: float = 0.9,
        drift_threshold: float = 0.6
    ):
        """
        Initialize the PsiKoopmanIntegrator.
        
        Args:
            basis_type: Type of basis functions for Koopman estimator
            n_eigenfunctions: Number of eigenfunctions to compute
            time_step: Time step between samples
            stable_threshold: Threshold for stable synchronization
            drift_threshold: Threshold for drifting synchronization
        """
        # Initialize Koopman estimator
        self.koopman_estimator = KoopmanEstimator(
            basis_type=basis_type,
            basis_params={"n_harmonics": 3} if basis_type == "fourier" else {"degree": 2},
            dt=time_step,
            n_eigenfunctions=n_eigenfunctions
        )
        
        # Initialize sync monitor
        self.sync_monitor = PsiSyncMonitor(
            stable_threshold=stable_threshold,
            drift_threshold=drift_threshold
        )
        
        # Storage for concept metadata
        self.concept_metadata: Dict[str, Any] = {}
        
        # Current state
        self.current_state: Optional[PsiPhaseState] = None
        self.current_metrics: Optional[PsiSyncMetrics] = None
        
        logger.info(
            f"PsiKoopmanIntegrator initialized with {n_eigenfunctions} eigenfunctions, "
            f"basis={basis_type}, thresholds: stable={stable_threshold}, drift={drift_threshold}"
        )
    
    def process_time_series(
        self, 
        time_series: np.ndarray,
        concept_ids: Optional[List[str]] = None
    ) -> Tuple[List[KoopmanEigenMode], PsiSyncMetrics]:
        """
        Process time series data to extract eigenfunctions and evaluate stability.
        
        Args:
            time_series: Time series data with shape (n_samples, n_features)
            concept_ids: Optional identifiers for concepts
            
        Returns:
            Tuple of (eigen_modes, sync_metrics)
        """
        # Fit Koopman model to time series
        self.koopman_estimator.fit(time_series)
        
        # Get eigenmodes
        eigen_modes = self.koopman_estimator.eigenmodes
        
        # Extract phase information from dominant eigenmode
        dominant_mode = self.koopman_estimator.get_dominant_mode()
        
        # Complex eigenfunction values contain phase information
        psi_values = np.array([mode.eigenfunction for mode in eigen_modes]).T
        
        # Extract phases from dominant eigenfunction
        # The phase is the argument of the complex eigenfunction
        phases = np.angle(dominant_mode.eigenfunction)
        
        # For simplified demo, create a coupling matrix based on eigenfunction similarity
        n_concepts = len(phases)
        coupling_matrix = np.zeros((n_concepts, n_concepts))
        
        # Simple heuristic: couple concepts with similar eigenfunction values
        for i in range(n_concepts):
            for j in range(n_concepts):
                if i != j:
                    # Compute similarity between eigenfunctions
                    psi_i = psi_values[i]
                    psi_j = psi_values[j]
                    
                    # Normalize
                    psi_i_norm = np.linalg.norm(psi_i)
                    psi_j_norm = np.linalg.norm(psi_j)
                    
                    if psi_i_norm > 0 and psi_j_norm > 0:
                        # Use abs of dot product as similarity
                        similarity = np.abs(np.vdot(psi_i, psi_j)) / (psi_i_norm * psi_j_norm)
                        coupling_matrix[i, j] = similarity
        
        # Create PsiPhaseState
        if concept_ids is None:
            concept_ids = [f"concept_{i}" for i in range(n_concepts)]
            
        state = PsiPhaseState(
            theta=phases,
            psi=psi_values,
            coupling_matrix=coupling_matrix,
            concept_ids=concept_ids
        )
        
        # Store current state
        self.current_state = state
        
        # Evaluate synchronization
        metrics = self.sync_monitor.evaluate(state)
        self.current_metrics = metrics
        
        logger.info(
            f"Processed time series with {n_concepts} concepts. "
            f"Synchrony: {metrics.synchrony_score:.2f}, "
            f"State: {metrics.sync_state.name}"
        )
        
        return eigen_modes, metrics
    
    def get_stability_recommendations(self) -> SyncAction:
        """
        Get recommendations based on current stability assessment.
        
        Returns:
            SyncAction with recommendations
        """
        if self.current_state is None or self.current_metrics is None:
            raise ValueError("No state available. Call process_time_series first.")
            
        return self.sync_monitor.recommend_action(self.current_metrics, self.current_state)
    
    def apply_coupling_adjustments(self) -> Optional[np.ndarray]:
        """
        Apply recommended coupling adjustments to improve stability.
        
        Returns:
            Updated coupling matrix or None if no state available
        """
        if self.current_state is None or self.current_metrics is None:
            return None
            
        # Get recommendations
        action = self.get_stability_recommendations()
        
        if action.coupling_adjustments is None:
            return None
            
        # Apply adjustments
        if self.current_state.coupling_matrix is not None:
            new_coupling = self.current_state.coupling_matrix + action.coupling_adjustments
            
            # Ensure positive coupling
            new_coupling = np.maximum(0.0, new_coupling)
            
            # Update state
            self.current_state.coupling_matrix = new_coupling
            
            logger.info(f"Applied coupling adjustments with average magnitude: {np.mean(np.abs(action.coupling_adjustments)):.4f}")
            
            return new_coupling
            
        return None
    
    def predict_concept_evolution(
        self,
        n_steps: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict how concepts will evolve over time using Koopman dynamics.
        
        Args:
            n_steps: Number of steps to predict
            
        Returns:
            Tuple of (predicted_phases, predicted_values)
        """
        if self.current_state is None:
            raise ValueError("No state available. Call process_time_series first.")
            
        # Use Koopman operator to predict future states
        n_concepts = len(self.current_state.theta)
        
        # We need some state representation for prediction
        # For simplicity, use the real part of psi as state
        current_state = np.mean(self.current_state.psi.real, axis=1)
        
        # Predict using Koopman operator
        predicted_values = np.zeros((n_steps, n_concepts))
        predicted_values[0] = current_state
        
        for i in range(1, n_steps):
            predicted_values[i] = self.koopman_estimator.predict(
                predicted_values[i-1].reshape(1, -1)
            ).flatten()
        
        # Also predict phases using simple phase oscillator update rule
        predicted_phases = np.zeros((n_steps, n_concepts))
        predicted_phases[0] = self.current_state.theta
        
        if self.current_state.coupling_matrix is not None:
            # Simple Kuramoto model for phase updates
            for i in range(1, n_steps):
                phases = predicted_phases[i-1].copy()
                phase_updates = np.zeros(n_concepts)
                
                for j in range(n_concepts):
                    for k in range(n_concepts):
                        if j != k:
                            phase_diff = phases[k] - phases[j]
                            # Wrap to [-π, π]
                            phase_diff = ((phase_diff + np.pi) % (2*np.pi)) - np.pi
                            phase_updates[j] += 0.1 * self.current_state.coupling_matrix[j, k] * np.sin(phase_diff)
                
                # Update phases
                predicted_phases[i] = (phases + phase_updates) % (2*np.pi)
        else:
            # No coupling, phases remain constant
            for i in range(1, n_steps):
                predicted_phases[i] = predicted_phases[0]
        
        return predicted_phases, predicted_values
    
    def plot_stability_assessment(self, title: str = "ψ-Sync Stability Assessment"):
        """
        Plot the current stability assessment.
        
        Args:
            title: Plot title
        """
        if self.current_state is None or self.current_metrics is None:
            raise ValueError("No state available. Call process_time_series first.")
            
        plt.figure(figsize=(15, 10))
        
        # 1. Plot phases on unit circle
        ax1 = plt.subplot(221, polar=True)
        
        # Plot unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax1.plot(theta, np.ones_like(theta), 'k--', alpha=0.3)
        
        # Plot oscillators
        n_concepts = len(self.current_state.theta)
        colors = plt.cm.viridis(np.linspace(0, 1, n_concepts))
        
        for i, phase in enumerate(self.current_state.theta):
            ax1.scatter(phase, 1.0, color=colors[i], s=100, label=self.current_state.concept_ids[i])
            
        ax1.set_rticks([])  # Hide radial ticks
        ax1.set_title("Phase Distribution")
        
        # Only show legend if not too many concepts
        if n_concepts <= 7:
            ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 2. Plot coupling matrix
        ax2 = plt.subplot(222)
        
        if self.current_state.coupling_matrix is not None:
            im = ax2.imshow(
                self.current_state.coupling_matrix, 
                cmap='viridis', 
                interpolation='nearest',
                vmin=0,
                vmax=1
            )
            plt.colorbar(im, ax=ax2, label="Coupling Strength")
            
            # Add concept labels
            concept_labels = self.current_state.concept_ids
            if len(concept_labels) <= 10:  # Only show labels if not too many
                ax2.set_xticks(np.arange(len(concept_labels)))
                ax2.set_yticks(np.arange(len(concept_labels)))
                ax2.set_xticklabels(concept_labels, rotation=45, ha="right")
                ax2.set_yticklabels(concept_labels)
                
            ax2.set_title("Coupling Matrix")
        else:
            ax2.text(0.5, 0.5, "No coupling matrix available", 
                    ha='center', va='center', fontsize=12)
            ax2.set_title("Coupling Matrix (Not Available)")
        
        # 3. Plot metrics
        ax3 = plt.subplot(223)
        
        metrics_data = {
            'Synchrony': self.current_metrics.synchrony_score,
            'Integrity': self.current_metrics.attractor_integrity,
            'Residual': self.current_metrics.residual_energy,
            'Lyapunov Δ': abs(self.current_metrics.lyapunov_delta) * 10  # Scale for visibility
        }
        
        # Create bars
        bars = ax3.bar(metrics_data.keys(), metrics_data.values())
        
        # Color by sync state
        if self.current_metrics.sync_state == SyncState.STABLE:
            bars[0].set_color('green')
            bars[1].set_color('green')
        elif self.current_metrics.sync_state == SyncState.DRIFT:
            bars[0].set_color('orange')
            bars[1].set_color('orange')
        else:
            bars[0].set_color('red')
            bars[1].set_color('red')
            
        # Use different color for residual
        bars[2].set_color('purple')
        
        # Set lyapunov delta color based on sign
        lyapunov_color = 'green' if self.current_metrics.lyapunov_delta <= 0 else 'red'
        bars[3].set_color(lyapunov_color)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom')
        
        ax3.set_ylim(0, 1.2)
        ax3.set_title("Synchronization Metrics")
        
        # 4. Plot eigenfunction projection
        ax4 = plt.subplot(224)
        
        # Extract real part of top 2 eigenfunctions for 2D projection
        if self.current_state.psi.shape[1] >= 2:
            x = self.current_state.psi[:, 0].real
            y = self.current_state.psi[:, 1].real
            
            # Plot points
            ax4.scatter(x, y, c=colors, s=100)
            
            # Add labels if not too many points
            if n_concepts <= 7:
                for i, (xi, yi) in enumerate(zip(x, y)):
                    ax4.text(xi, yi, self.current_state.concept_ids[i], fontsize=9)
                    
            # Draw lines between points based on coupling
            if self.current_state.coupling_matrix is not None:
                for i in range(n_concepts):
                    for j in range(i+1, n_concepts):
                        # Only draw strong connections
                        coupling = self.current_state.coupling_matrix[i, j]
                        if coupling > 0.3:  # Arbitrary threshold
                            ax4.plot(
                                [x[i], x[j]], [y[i], y[j]], 
                                'k-', alpha=min(1.0, coupling),
                                linewidth=coupling*2
                            )
            
            ax4.set_xlabel("ψ₁ (Real Part)")
            ax4.set_ylabel("ψ₂ (Real Part)")
            ax4.set_title("Eigenfunction Projection")
            
            # Add grid
            ax4.grid(True, linestyle='--', alpha=0.7)
            
            # Equal aspect ratio for clearer visualization
            ax4.set_aspect('equal', adjustable='box')
        else:
            ax4.text(0.5, 0.5, "Need at least 2 eigenfunctions for projection", 
                    ha='center', va='center', fontsize=12)
        
        # Add state as text
        state_colors = {
            SyncState.STABLE: 'green',
            SyncState.DRIFT: 'orange',
            SyncState.BREAK: 'red',
            SyncState.UNKNOWN: 'gray'
        }
        
        # Get recommendation
        action = self.get_stability_recommendations()
        
        plt.figtext(
            0.5, 0.01, 
            f"State: {self.current_metrics.sync_state.name} | Confidence: {action.confidence:.2f}", 
            ha='center', 
            color=state_colors[self.current_metrics.sync_state], 
            fontsize=14, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
        )
        
        # Add recommendation as subtitle
        plt.figtext(
            0.5, 0.04,
            f"Recommendation: {action.recommendation}",
            ha='center',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
        )
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        plt.show()
        
    def plot_predicted_evolution(self, n_steps: int = 10):
        """
        Plot the predicted evolution of concepts.
        
        Args:
            n_steps: Number of steps to predict
        """
        if self.current_state is None:
            raise ValueError("No state available. Call process_time_series first.")
            
        # Predict evolution
        predicted_phases, predicted_values = self.predict_concept_evolution(n_steps)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # 1. Plot phase evolution
        ax1 = plt.subplot(211)
        
        n_concepts = len(self.current_state.theta)
        colors = plt.cm.viridis(np.linspace(0, 1, n_concepts))
        
        for i in range(n_concepts):
            ax1.plot(
                np.arange(n_steps), 
                predicted_phases[:, i], 
                'o-', 
                color=colors[i],
                label=self.current_state.concept_ids[i] if n_concepts <= 7 else None
            )
            
        ax1.set_ylim(0, 2*np.pi)
        ax1.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax1.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Phase (θ)")
        ax1.set_title("Predicted Phase Evolution")
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        if n_concepts <= 7:
            ax1.legend(loc='upper right')
        
        # 2. Plot value evolution
        ax2 = plt.subplot(212)
        
        for i in range(n_concepts):
            ax2.plot(
                np.arange(n_steps), 
                predicted_values[:, i], 
                'o-', 
                color=colors[i],
                label=self.current_state.concept_ids[i] if n_concepts <= 7 else None
            )
            
        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("Concept Value")
        ax2.set_title("Predicted Concept Evolution")
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        if n_concepts <= 7:
            ax2.legend(loc='upper right')
        
        plt.suptitle("ψ-Driven Concept Evolution Prediction", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def generate_synthetic_time_series(
    n_samples: int = 100,
    n_concepts: int = 5,
    oscillation_periods: List[float] = [10, 20, 30],
    noise_level: float = 0.1,
    trend_strength: float = 0.1
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate synthetic time series data for testing.
    
    Args:
        n_samples: Number of time steps to generate
        n_concepts: Number of concepts (time series variables)
        oscillation_periods: List of periods for oscillatory components
        noise_level: Level of noise to add
        trend_strength: Strength of trend component
        
    Returns:
        Tuple of (time_series, concept_ids)
    """
    # Time vector
    t = np.arange(n_samples)
    
    # Initialize time series
    time_series = np.zeros((n_samples, n_concepts))
    
    # Concept names
    concept_names = [
        "Memory", "Attention", "Reasoning", "Perception",
        "Learning", "Creativity", "Emotion", "Planning",
        "Language", "Decision"
    ]
    
    # Ensure we have enough names
    if n_concepts > len(concept_names):
        # Add more generic names
        for i in range(len(concept_names), n_concepts):
            concept_names.append(f"Concept_{i+1}")
    
    # Truncate to requested number
    concept_ids = concept_names[:n_concepts]
    
    # Generate data for each concept
    for i in range(n_concepts):
        # Base signal: sum of oscillations with different periods
        signal = np.zeros(n_samples)
        
        for period in oscillation_periods:
            # Add each oscillatory component with random phase
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.uniform(0.5, 1.5)
            signal += amplitude * np.sin(2*np.pi*t/period + phase)
        
        # Add trend
        trend = trend_strength * t / n_samples
        
        # Add noise
        noise = np.random.normal(0, noise_level, n_samples)
        
        # Combine components
        time_series[:, i] = signal + trend + noise
    
    return time_series, concept_ids

def run_demo():
    """Run a demonstration of the PsiKoopmanIntegrator."""
    print("\n=== ψ-Sync + Koopman Integration Demo ===\n")
    
    # Create integrator
    integrator = PsiKoopmanIntegrator(
        basis_type="fourier",
        n_eigenfunctions=3,
        stable_threshold=0.85,
        drift_threshold=0.6
    )
    
    # Generate synthetic data
    print("Generating synthetic time series data...")
    time_series, concept_ids = generate_synthetic_time_series(
        n_samples=100,
        n_concepts=5,
        oscillation_periods=[10, 20, 30],
        noise_level=0.2
    )
    
    # Process data
    print("\nProcessing time series with Koopman estimator and ψ-Sync monitor...")
    eigen_modes, metrics = integrator.process_time_series(time_series, concept_ids)
    
    # Print summary
    print(f"\nFound {len(eigen_modes)} eigenmodes")
    print(f"Dominant mode frequency: {eigen_modes[0].frequency:.4f}")
    print(f"Synchrony score: {metrics.synchrony_score:.2f}")
    print(f"Attractor integrity: {metrics.attractor_integrity:.2f}")
    print(f"Stability state: {metrics.sync_state.name}")
    
    # Get recommendations
    action = integrator.get_stability_recommendations()
    print(f"\nRecommendation: {action.recommendation}")
    print(f"Confidence: {action.confidence:.2f}")
    print(f"Requires user confirmation: {action.requires_user_confirmation}")
    
    # Plot stability assessment
    print("\nPlotting stability assessment...")
    integrator.plot_stability_assessment("Initial ψ-Sync Stability Assessment")
    
    # Apply coupling adjustments if needed
    if metrics.sync_state != SyncState.STABLE:
        print("\nApplying coupling adjustments to improve stability...")
        new_coupling = integrator.apply_coupling_adjustments()
        
        if new_coupling is not None:
            # Re-evaluate
            new_metrics = integrator.sync_monitor.evaluate(integrator.current_state)
            
            print(f"After adjustment - Synchrony: {new_metrics.synchrony_score:.2f}")
            print(f"After adjustment - State: {new_metrics.sync_state.name}")
            
            # Plot updated assessment
            integrator.plot_stability_assessment("After Coupling Adjustment")
    
    # Plot predicted evolution
    print("\nPlotting predicted concept evolution...")
    integrator.plot_predicted_evolution(n_steps=15)
    
    print("\n=== End of Demo ===")

if __name__ == "__main__":
    run_demo()
