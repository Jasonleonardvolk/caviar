"""
ELFIN DSL Stability Integration Demo (Direct Version).

This script demonstrates the integration of ELFIN DSL with ψ-Sync stability monitoring
and Lyapunov verification, using the direct API without requiring the parser/compiler.

This version bypasses the need for the full ELFIN parser and compiler machinery,
focusing only on the stability aspects.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("elfin.examples.stability_direct")

# Import ELFIN stability components directly
from alan_backend.elfin.stability.lyapunov import (
    LyapunovFunction,
    PolynomialLyapunov,
    NeuralLyapunov,
    CLVFunction,
    CompositeLyapunov,
    ProofStatus
)

from alan_backend.elfin.stability.verifier import (
    LyapunovVerifier,
    ProofCache,
    ConstraintIR
)

from alan_backend.elfin.stability.psi_bridge import (
    PsiConceptBridge,
    PhaseStateUpdate,
    ConceptPhaseMapping
)

# Import ψ-Sync components
from alan_backend.banksy import (
    PsiSyncMonitor,
    PsiPhaseState,
    PsiSyncMetrics,
    SyncAction,
    SyncState
)

def create_test_phase_state(n_concepts: int = 5, n_modes: int = 3, coherence: float = 0.8) -> PsiPhaseState:
    """Create a test phase state for demonstration.
    
    Args:
        n_concepts: Number of concepts/oscillators
        n_modes: Number of ψ-modes
        coherence: Phase coherence level (0-1)
        
    Returns:
        A PsiPhaseState for testing
    """
    # Generate phases with some coherence
    if coherence > 0.9:
        # High coherence - similar phases
        mean_phase = np.random.uniform(0, 2*np.pi)
        phases = mean_phase + np.random.normal(0, 0.2, n_concepts)
    elif coherence > 0.6:
        # Medium coherence - a few clusters
        n_clusters = 2
        cluster_size = n_concepts // n_clusters
        phases = np.zeros(n_concepts)
        
        for i in range(n_clusters):
            mean_phase = np.random.uniform(0, 2*np.pi)
            start_idx = i * cluster_size
            end_idx = min(start_idx + cluster_size, n_concepts)
            phases[start_idx:end_idx] = mean_phase + np.random.normal(0, 0.3, end_idx - start_idx)
    else:
        # Low coherence - random phases
        phases = np.random.uniform(0, 2*np.pi, n_concepts)
        
    # Ensure phases are in [0, 2π)
    phases = phases % (2 * np.pi)
    
    # Generate ψ values
    psi = np.random.normal(0, 1, (n_concepts, n_modes))
    
    # Generate coupling matrix
    coupling_matrix = np.zeros((n_concepts, n_concepts))
    for i in range(n_concepts):
        for j in range(n_concepts):
            if i != j:
                # Base coupling on phase similarity
                phase_diff = np.abs(phases[i] - phases[j])
                phase_diff = min(phase_diff, 2*np.pi - phase_diff)
                coupling_matrix[i, j] = 0.5 * np.exp(-phase_diff)
                
    # Generate concept IDs
    concept_ids = [f"concept_{i}" for i in range(n_concepts)]
    
    return PsiPhaseState(
        theta=phases,
        psi=psi,
        coupling_matrix=coupling_matrix,
        concept_ids=concept_ids
    )

def create_polynomial_lyapunov() -> PolynomialLyapunov:
    """Create a polynomial Lyapunov function for demonstration.
    
    Returns:
        A polynomial Lyapunov function
    """
    # Create a positive definite Q matrix
    # For a quadratic Lyapunov function V(x) = x^T Q x
    dim = 3  # State dimension (theta + 2 psi modes)
    
    # Start with a random matrix
    A = np.random.normal(0, 1, (dim, dim))
    
    # Make it symmetric positive definite
    Q = A @ A.T + np.eye(dim) * 2.0
    
    # Create the Lyapunov function
    return PolynomialLyapunov(
        name="poly_lyap",
        q_matrix=Q,
        basis_functions=[f"x{i}" for i in range(dim)],
        domain_concept_ids=["concept_0", "concept_1"]
    )

def create_neural_lyapunov() -> NeuralLyapunov:
    """Create a neural Lyapunov function for demonstration.
    
    Returns:
        A neural Lyapunov function
    """
    # Define a simple neural network architecture
    layer_dims = [3, 10, 5, 1]  # 3 inputs, 10 hidden, 5 hidden, 1 output
    
    # Create random weights for demonstration
    weights = []
    for i in range(len(layer_dims) - 1):
        in_dim = layer_dims[i]
        out_dim = layer_dims[i + 1]
        
        # Create random weights and biases
        W = np.random.normal(0, 1/np.sqrt(in_dim), (in_dim, out_dim))
        b = np.zeros(out_dim)
        
        weights.append((W, b))
    
    # Create the Lyapunov function
    return NeuralLyapunov(
        name="neural_lyap",
        layer_dims=layer_dims,
        weights=weights,
        input_bounds=[(-np.pi, np.pi), (-2, 2), (-2, 2)],
        domain_concept_ids=["concept_2", "concept_3"]
    )

def create_clf_function() -> CLVFunction:
    """Create a Control Lyapunov-Value Function for demonstration.
    
    Returns:
        A CLF function
    """
    # Define a simple quadratic value function
    def value_fn(x):
        return np.sum(x**2)
    
    # Create the CLF
    return CLVFunction(
        name="control_lyap",
        value_function=value_fn,
        control_variables=["u1", "u2"],
        clf_gamma=0.1,
        domain_concept_ids=["concept_3", "concept_4"]
    )

def create_composite_lyapunov(components: List[LyapunovFunction]) -> CompositeLyapunov:
    """Create a composite Lyapunov function for demonstration.
    
    Args:
        components: Component Lyapunov functions
        
    Returns:
        A composite Lyapunov function
    """
    return CompositeLyapunov(
        name="composite_lyap",
        component_lyapunovs=components,
        weights=[1.0, 0.5, 0.3],  # Weight each component differently
        composition_type="weighted_sum",
        domain_concept_ids=["concept_0", "concept_2", "concept_4"]
    )

def simulate_phase_dynamics(
    state: PsiPhaseState,
    steps: int = 10, 
    dt: float = 0.1,
    noise_level: float = 0.02
) -> List[PsiPhaseState]:
    """Simulate phase dynamics for a series of steps.
    
    Args:
        state: Initial phase state
        steps: Number of simulation steps
        dt: Time step
        noise_level: Level of noise to add
        
    Returns:
        List of phase states from simulation
    """
    states = [state]
    
    n_oscillators = len(state.theta)
    
    for _ in range(steps):
        # Get the latest state
        current = states[-1]
        
        # Create a copy of the current state
        new_theta = current.theta.copy()
        new_psi = current.psi.copy()
        
        # Update phases based on Kuramoto model
        for i in range(n_oscillators):
            # Phase update due to coupling
            phase_update = 0.0
            
            if current.coupling_matrix is not None:
                for j in range(n_oscillators):
                    if i != j:
                        # Compute phase difference
                        phase_diff = current.theta[j] - current.theta[i]
                        # Add coupling effect
                        coupling = current.coupling_matrix[i, j]
                        phase_update += coupling * np.sin(phase_diff)
            
            # Apply phase update
            new_theta[i] += dt * phase_update
            
            # Add some noise
            new_theta[i] += np.random.normal(0, noise_level)
            
            # Ensure phase is in [0, 2π)
            new_theta[i] = new_theta[i] % (2 * np.pi)
            
        # Simple update for ψ values - just add some noise
        new_psi += np.random.normal(0, noise_level, new_psi.shape)
        
        # Create new state
        new_state = PsiPhaseState(
            theta=new_theta,
            psi=new_psi,
            coupling_matrix=current.coupling_matrix,
            concept_ids=current.concept_ids
        )
        
        # Add to list
        states.append(new_state)
        
    return states

def plot_stability_results(
    states: List[PsiPhaseState],
    bridge: PsiConceptBridge,
    lyapunov_fns: List[LyapunovFunction]
):
    """Plot simulation results with stability analysis.
    
    Args:
        states: Phase states from simulation
        bridge: PsiConceptBridge with stability monitoring
        lyapunov_fns: Lyapunov functions to monitor
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Phase trajectories
    ax1 = plt.subplot(2, 2, 1)
    
    n_oscillators = len(states[0].theta)
    for i in range(n_oscillators):
        phases = [state.theta[i] for state in states]
        ax1.plot(phases, label=f"Oscillator {i}")
        
    ax1.set_title("Phase Trajectories")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Phase (radians)")
    ax1.legend()
    
    # Plot 2: Synchrony metrics
    ax2 = plt.subplot(2, 2, 2)
    
    synchrony_scores = [
        bridge.monitor.evaluate(state).synchrony_score
        for state in states
    ]
    
    integrity_scores = [
        bridge.monitor.evaluate(state).attractor_integrity
        for state in states
    ]
    
    ax2.plot(synchrony_scores, label="Synchrony", color='blue')
    ax2.plot(integrity_scores, label="Attractor Integrity", color='green')
    ax2.axhline(y=bridge.synchrony_threshold, color='red', linestyle='--', label="Threshold")
    
    ax2.set_title("Synchronization Metrics")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    
    # Plot 3: Lyapunov values
    ax3 = plt.subplot(2, 2, 3)
    
    # Evaluate Lyapunov functions for each state
    lyapunov_values = {}
    
    for lyap_fn in lyapunov_fns:
        values = []
        for state in states:
            # We'll use the first concept's state for simplicity
            concept_id = lyap_fn.domain_concept_ids[0]
            mapping = bridge.concept_to_phase.get(concept_id)
            
            if mapping is not None and mapping.phase_index < len(state.theta):
                theta = state.theta[mapping.phase_index]
                psi = state.psi[mapping.phase_index]
                x = np.concatenate(([theta], psi.flatten()))
                values.append(lyap_fn.evaluate(x))
            else:
                values.append(np.nan)
                
        lyapunov_values[lyap_fn.name] = values
        ax3.plot(values, label=lyap_fn.name)
        
    ax3.set_title("Lyapunov Function Values")
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Value")
    ax3.legend()
    
    # Plot 4: Phase space visualization (first 2 oscillators)
    ax4 = plt.subplot(2, 2, 4, polar=True)
    
    # Plot the final state's phases on the unit circle
    final_state = states[-1]
    
    # Plot unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax4.plot(theta, np.ones_like(theta), 'k--', alpha=0.3)
    
    # Plot oscillators
    colors = plt.cm.viridis(np.linspace(0, 1, n_oscillators))
    for i in range(n_oscillators):
        ax4.scatter(final_state.theta[i], 1.0, color=colors[i], s=100, label=f"Osc {i}")
        
    ax4.set_rticks([])  # Hide radial ticks
    ax4.set_title("Final Phase Distribution")
    
    plt.tight_layout()
    plt.show()

def run_stability_demo():
    """Run the ELFIN stability demo."""
    logger.info("Starting ELFIN Stability Demo (Direct Version)")
    
    # Step 1: Create a ψ-Sync monitor
    monitor = PsiSyncMonitor(
        stable_threshold=0.9,
        drift_threshold=0.6,
    )
    logger.info("Created PsiSyncMonitor")
    
    # Step 2: Create a ψ-Concept bridge
    bridge = PsiConceptBridge(
        psi_sync_monitor=monitor,
        synchrony_threshold=0.8
    )
    logger.info("Created PsiConceptBridge")
    
    # Step 3: Create initial phase state
    state = create_test_phase_state(n_concepts=5, n_modes=3, coherence=0.8)
    logger.info(f"Created initial phase state with {len(state.theta)} oscillators")
    
    # Step 4: Register concepts with the bridge
    for i, concept_id in enumerate(state.concept_ids):
        bridge.register_concept(
            concept_id=concept_id,
            phase_index=i,
            psi_mode_indices=[0, 1, 2],  # Use all 3 modes
            psi_mode_weights=[1.0, 0.5, 0.25]  # Weight them differently
        )
    logger.info(f"Registered {len(state.concept_ids)} concepts with bridge")
    
    # Step 5: Create Lyapunov functions
    poly_lyap = create_polynomial_lyapunov()
    neural_lyap = create_neural_lyapunov()
    clf = create_clf_function()
    
    # Register with bridge
    bridge.register_lyapunov_function(poly_lyap)
    bridge.register_lyapunov_function(neural_lyap)
    bridge.register_lyapunov_function(clf)
    
    # Create composite Lyapunov function
    composite_lyap = create_composite_lyapunov([poly_lyap, neural_lyap, clf])
    bridge.register_lyapunov_function(composite_lyap)
    
    logger.info("Created and registered 4 Lyapunov functions")
    
    # Step 6: Create a Lyapunov verifier
    verifier = LyapunovVerifier()
    
    # Verify polynomial Lyapunov function
    result = verifier.verify(poly_lyap)
    logger.info(f"Polynomial Lyapunov verification: {result.status}")
    
    # Step 7: Run simulation
    logger.info("Running phase dynamics simulation...")
    states = simulate_phase_dynamics(state, steps=50)
    
    # Step 8: Process states through bridge
    for state in states:
        bridge.update_phase_state(state)
    
    # Step 9: Check stability status
    for concept_id in state.concept_ids:
        status = bridge.get_concept_stability_status(concept_id)
        logger.info(f"Concept {concept_id} stability: {status['sync_status']}, {status['lyapunov_status']}")
        
    # Step 10: Demo transition verification
    transition_valid = bridge.verify_transition(
        from_concept_id="concept_0",
        to_concept_id="concept_1",
        composite_lyapunov=composite_lyap
    )
    logger.info(f"Transition from concept_0 to concept_1 is {'valid' if transition_valid else 'invalid'}")
    
    # Step 11: Get coupling recommendations
    coupling_adj = bridge.recommend_coupling_adjustments()
    if coupling_adj is not None:
        logger.info(f"Recommended coupling adjustments matrix shape: {coupling_adj.shape}")
    
    # Step 12: Plot results
    logger.info("Plotting results")
    plot_stability_results(states, bridge, [poly_lyap, neural_lyap, clf, composite_lyap])
    
    logger.info("ELFIN Stability Demo completed")

if __name__ == "__main__":
    run_stability_demo()
