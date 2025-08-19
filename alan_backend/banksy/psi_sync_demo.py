"""
Demo of the PsiSyncMonitor functionality for phase-eigenfunction synchronization.

This script demonstrates how to:
1. Create a PsiSyncMonitor
2. Prepare phase and eigenfunction data
3. Evaluate synchronization state
4. Get recommended actions based on stability assessment
5. Apply coupling adjustments to improve synchronization

It can be run directly to see a visual example of the ψ-Sync system in action.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import our modules
from alan_backend.banksy import (
    PsiSyncMonitor, 
    PsiPhaseState, 
    PsiSyncMetrics, 
    SyncAction,
    SyncState
)

def generate_test_data(
    n_concepts: int = 10, 
    n_modes: int = 3,
    coherence: float = 0.7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate test data for phase oscillators and eigenfunctions.
    
    Args:
        n_concepts: Number of concepts (oscillators)
        n_modes: Number of Koopman modes
        coherence: Desired coherence level (0-1)
        
    Returns:
        Tuple of (phases, psi_values, coupling_matrix)
    """
    # Generate phases with desired coherence
    if coherence > 0.9:
        # High coherence - all phases close together
        mean_phase = np.random.uniform(0, 2*np.pi)
        spread = 0.1  # Small spread for high coherence
        phases = mean_phase + np.random.normal(0, spread, n_concepts)
    elif coherence > 0.6:
        # Medium coherence - two or three clusters
        n_clusters = 2
        phases = np.zeros(n_concepts)
        cluster_size = n_concepts // n_clusters
        for i in range(n_clusters):
            mean_phase = np.random.uniform(0, 2*np.pi)
            spread = 0.3  # Medium spread
            cluster_phases = mean_phase + np.random.normal(0, spread, cluster_size)
            start_idx = i * cluster_size
            end_idx = start_idx + cluster_size
            phases[start_idx:end_idx] = cluster_phases
            
        # Handle remaining concepts
        if n_concepts % n_clusters != 0:
            remaining = n_concepts - (n_clusters * cluster_size)
            phases[-remaining:] = np.random.uniform(0, 2*np.pi, remaining)
    else:
        # Low coherence - random phases
        phases = np.random.uniform(0, 2*np.pi, n_concepts)
    
    # Keep phases in [0, 2π)
    phases = phases % (2 * np.pi)
    
    # Generate eigenfunction values
    # Usually complex, but for simplicity we'll use real values
    psi_values = np.random.normal(0, 1, (n_concepts, n_modes))
    
    # Generate coupling matrix
    coupling_matrix = np.zeros((n_concepts, n_concepts))
    
    # Add some coupling between nearby concepts
    for i in range(n_concepts):
        for j in range(n_concepts):
            if i != j:
                # Stronger coupling for oscillators with similar phases
                phase_diff = abs(phases[i] - phases[j])
                phase_diff = min(phase_diff, 2*np.pi - phase_diff)  # Shortest path
                
                # Coupling strength inversely proportional to phase difference
                coupling_matrix[i, j] = 0.5 * np.exp(-phase_diff)
    
    return phases, psi_values, coupling_matrix

def plot_phases_and_metrics(
    phases: np.ndarray, 
    metrics: PsiSyncMetrics,
    title: str = "Phase Oscillator State"
):
    """
    Plot the phase oscillators on a unit circle and show metrics.
    
    Args:
        phases: Array of oscillator phases
        metrics: Current PsiSyncMetrics
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Plot phases on unit circle
    ax1 = plt.subplot(121, polar=True)
    
    # Plot unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(theta, np.ones_like(theta), 'k--', alpha=0.3)
    
    # Plot oscillators
    colors = plt.cm.viridis(np.linspace(0, 1, len(phases)))
    for i, phase in enumerate(phases):
        ax1.scatter(phase, 1.0, color=colors[i], s=100)
        
    ax1.set_rticks([])  # Hide radial ticks
    ax1.set_title("Phase Distribution")
    
    # Plot metrics
    ax2 = plt.subplot(122)
    
    metrics_data = {
        'Synchrony': metrics.synchrony_score,
        'Integrity': metrics.attractor_integrity,
        'Residual': metrics.residual_energy,
        'Lyapunov Δ': abs(metrics.lyapunov_delta) * 10  # Scale for visibility
    }
    
    # Create bars
    bars = ax2.bar(metrics_data.keys(), metrics_data.values())
    
    # Color by sync state
    if metrics.sync_state == SyncState.STABLE:
        bars[0].set_color('green')
        bars[1].set_color('green')
    elif metrics.sync_state == SyncState.DRIFT:
        bars[0].set_color('orange')
        bars[1].set_color('orange')
    else:
        bars[0].set_color('red')
        bars[1].set_color('red')
        
    # Use different color for residual
    bars[2].set_color('purple')
    
    # Set lyapunov delta color based on sign
    lyapunov_color = 'green' if metrics.lyapunov_delta <= 0 else 'red'
    bars[3].set_color(lyapunov_color)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom')
    
    ax2.set_ylim(0, 1.2)
    ax2.set_title("Synchronization Metrics")
    
    # Add state as text
    state_colors = {
        SyncState.STABLE: 'green',
        SyncState.DRIFT: 'orange',
        SyncState.BREAK: 'red',
        SyncState.UNKNOWN: 'gray'
    }
    
    plt.figtext(
        0.5, 0.01, 
        f"State: {metrics.sync_state.name}", 
        ha='center', 
        color=state_colors[metrics.sync_state], 
        fontsize=14, 
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
    )
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def run_test_scenario():
    """
    Run a test scenario demonstrating the PsiSyncMonitor functionality.
    """
    # Create a PsiSyncMonitor
    monitor = PsiSyncMonitor(
        stable_threshold=0.9,
        drift_threshold=0.6,
        residual_threshold=0.3
    )
    
    # Generate test data for different coherence levels
    coherence_levels = [0.4, 0.7, 0.95]
    
    for i, coherence in enumerate(coherence_levels):
        print(f"\n=== Testing with coherence level: {coherence:.2f} ===")
        
        # Generate data
        phases, psi_values, coupling_matrix = generate_test_data(
            n_concepts=10, n_modes=3, coherence=coherence
        )
        
        # Create state
        state = PsiPhaseState(
            theta=phases,
            psi=psi_values,
            coupling_matrix=coupling_matrix,
            concept_ids=[f"concept_{j}" for j in range(len(phases))]
        )
        
        # Evaluate state
        metrics = monitor.evaluate(state)
        
        print(f"Synchrony score: {metrics.synchrony_score:.2f}")
        print(f"Attractor integrity: {metrics.attractor_integrity:.2f}")
        print(f"Sync state: {metrics.sync_state.name}")
        
        # Get recommendations
        action = monitor.recommend_action(metrics, state)
        
        print(f"Recommendation: {action.recommendation}")
        print(f"Confidence: {action.confidence:.2f}")
        print(f"Requires confirmation: {action.requires_user_confirmation}")
        
        # Plot
        plot_phases_and_metrics(phases, metrics, f"Coherence Level: {coherence:.2f}")
        
        if i < len(coherence_levels) - 1:
            print("\nSimulating coupling adjustment...")
            
            # Apply coupling adjustments if any
            if action.coupling_adjustments is not None:
                # Update coupling matrix
                new_coupling = coupling_matrix + action.coupling_adjustments
                
                # Ensure positive coupling
                new_coupling = np.maximum(0.0, new_coupling)
                
                # Generate slightly adjusted phases based on new coupling
                adjusted_phases = phases.copy()
                for _ in range(10):  # Simulate a few update steps
                    for j in range(len(phases)):
                        # Simple update rule
                        phase_update = 0
                        for k in range(len(phases)):
                            if j != k:
                                phase_diff = phases[k] - phases[j]
                                # Wrap to [-π, π]
                                phase_diff = (phase_diff + np.pi) % (2*np.pi) - np.pi
                                phase_update += 0.1 * new_coupling[j, k] * np.sin(phase_diff)
                        adjusted_phases[j] += phase_update
                    
                adjusted_phases = adjusted_phases % (2*np.pi)
                
                # Create new state
                new_state = PsiPhaseState(
                    theta=adjusted_phases,
                    psi=psi_values,  # Keep same psi for simplicity
                    coupling_matrix=new_coupling,
                    concept_ids=[f"concept_{j}" for j in range(len(phases))]
                )
                
                # Evaluate new state
                new_metrics = monitor.evaluate(new_state)
                
                print(f"After adjustment - Synchrony: {new_metrics.synchrony_score:.2f}")
                print(f"After adjustment - State: {new_metrics.sync_state.name}")
                
                # Plot new state
                plot_phases_and_metrics(
                    adjusted_phases, 
                    new_metrics, 
                    f"After Coupling Adjustment (Coherence: {coherence:.2f})"
                )

if __name__ == "__main__":
    run_test_scenario()
