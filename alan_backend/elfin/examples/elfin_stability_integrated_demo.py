#!/usr/bin/env python3
"""
ELFIN DSL with Deep ψ-Sync Integration Demo.

This script demonstrates all components of the ELFIN stability framework:
- Koopman-based Lyapunov function learning
- Incremental verification with dependency tracking
- JIT-compiled stability enforcement
- Phase drift monitoring and adaptive reactions
"""

import os
import sys
import logging
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path to import ELFIN modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

try:
    # Import stability modules
    from alan_backend.elfin.stability.lyapunov import LyapunovFunction
    from alan_backend.elfin.stability.verifier import ProofStatus, LyapunovVerifier
    from alan_backend.elfin.stability.koopman_bridge_poc import (
        KoopmanLyapunov, create_koopman_lyapunov, generate_trajectory_data
    )
    from alan_backend.elfin.stability.incremental_verifier import (
        ParallelVerifier, ProofCache, DepGraph, VerificationResult
    )
    from alan_backend.elfin.stability.jit_guard import (
        StabilityGuard, CLFQuadraticProgramSolver, HAVE_NUMBA
    )
    from alan_backend.elfin.stability.phase_drift_monitor import (
        PhaseDriftMonitor, DriftThresholdType, AdaptiveActionType
    )
    
    HAVE_STABILITY = True
except ImportError:
    logger.warning("Could not import ELFIN stability modules. Using minimal implementations.")
    
    # Enum for proof status
    from enum import Enum, auto
    class ProofStatus(Enum):
        UNKNOWN = auto()
        VERIFIED = auto()
        REFUTED = auto()
        TIMEOUT = auto()
        ERROR = auto()
    
    # Minimal implementations for demos
    class LyapunovFunction:
        def __init__(self, name, domain_ids=None):
            self.name = name
            self.domain_ids = domain_ids or []
            
        def evaluate(self, x):
            return float(np.sum(x**2))
    
    class LyapunovVerifier:
        def verify(self, lyap, dynamics_fn=None):
            return ProofStatus.UNKNOWN
    
    class KoopmanLyapunov(LyapunovFunction):
        def __init__(self, name, lam, V, dict_fn, stable_mask, domain_ids=None):
            super().__init__(name, domain_ids)
    
    def create_koopman_lyapunov(states, next_states, dict_type="monomial", dict_params=None, 
                              name="V_koop", domain_ids=None, discrete_time=True):
        return KoopmanLyapunov(name, None, None, None, None, domain_ids)
    
    def generate_trajectory_data(dyn_fn, x0, n_steps, noise_scale=0.0):
        states = [x0]
        next_states = [dyn_fn(x0)]
        return states, next_states
    
    class ParallelVerifier:
        def __init__(self, verifier, cache=None, max_workers=None, timeout=300.0):
            self.verifier = verifier
    
    class ProofCache:
        def __init__(self, cache_dir=None):
            self.dep_graph = DepGraph()
    
    class DepGraph:
        def __init__(self):
            pass
    
    class VerificationResult:
        def __init__(self, status, lyapunov_name=None, certificate=None,
                   counterexample=None, verification_time=0.0, error_message=None):
            self.status = status
    
    class StabilityGuard:
        def __init__(self, lyap, threshold=0.0, callback=None, use_jit=True):
            self.lyap = lyap
    
    class CLFQuadraticProgramSolver:
        def __init__(self, lyap, control_dim, gamma=0.1, relaxation_weight=100.0, use_jit=True):
            self.lyap = lyap
    
    class PhaseDriftMonitor:
        def __init__(self, concept_to_psi_map, thresholds=None, banksy_monitor=None):
            self.concept_to_psi = concept_to_psi_map
    
    class DriftThresholdType(Enum):
        RADIANS = auto()
        PI_RATIO = auto()
        PERCENTAGE = auto()
        STANDARD_DEV = auto()
    
    class AdaptiveActionType(Enum):
        NOTIFY = auto()
        ADAPT_PLAN = auto()
        EXECUTE_AGENT = auto()
        CUSTOM_ACTION = auto()
    
    HAVE_STABILITY = False
    HAVE_NUMBA = False


# Example dynamical systems for the demo

def linear_system(x):
    """Linear system: dx/dt = Ax."""
    A = np.array([
        [0.9, 0.2],
        [-0.1, 0.8]
    ])
    return A @ x


def van_der_pol(x, mu=1.0, dt=0.1):
    """Van der Pol oscillator with Euler integration."""
    dx1 = x[1]
    dx2 = mu * (1 - x[0]**2) * x[1] - x[0]
    return np.array([x[0] + dt * dx1, x[1] + dt * dx2])


def controlled_system(x, u):
    """Controllable linear system: dx/dt = Ax + Bu."""
    A = np.array([
        [0.9, 0.2],
        [-0.1, 0.8]
    ])
    B = np.array([
        [1.0],
        [0.5]
    ])
    return A @ x + B @ u


# Example polynomial Lyapunov function
class QuadraticLyapunov(LyapunovFunction):
    """Quadratic Lyapunov function: V(x) = x^T Q x."""
    
    def __init__(self, name, Q, domain_ids=None):
        """
        Initialize quadratic Lyapunov function.
        
        Args:
            name: Name of the function
            Q: Quadratic form matrix
            domain_ids: IDs of concepts in the domain
        """
        super().__init__(name, domain_ids)
        self.Q = Q
        
    def evaluate(self, x):
        """Evaluate the function at x."""
        x = np.asarray(x)
        return float(x.T @ self.Q @ x)
    
    def get_quadratic_form(self):
        """Get the quadratic form matrix Q."""
        return self.Q
    
    def verify_positive_definite(self):
        """Verify that V(x) > 0 for all x ≠ 0."""
        # Positive definite if all eigenvalues of Q are positive
        eigenvalues = np.linalg.eigvalsh(self.Q)
        if np.all(eigenvalues > 0):
            return ProofStatus.VERIFIED
        return ProofStatus.REFUTED
    
    def verify_decreasing(self, dynamics_fn):
        """Verify that dV/dt < 0 for all x ≠ 0."""
        # For sophisticated verification, we'd use SOS programming
        # This is a simplistic check for demo purposes
        test_points = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.5, 0.5]),
            np.array([-0.7, 0.3])
        ]
        
        for x in test_points:
            if x[0] == 0 and x[1] == 0:
                continue
                
            # Compute gradient
            grad = 2 * self.Q @ x
            
            # Compute dynamics
            f_x = dynamics_fn(x)
            
            # Compute Lie derivative dV/dt = ∇V(x)·f(x)
            lie_derivative = np.dot(grad, f_x)
            
            if lie_derivative >= 0:
                return ProofStatus.REFUTED
        
        return ProofStatus.VERIFIED


# Simple verifier for the demo
class SimpleVerifier(LyapunovVerifier):
    """Simple Lyapunov verifier for demonstration purposes."""
    
    def verify(self, lyap, dynamics_fn=None):
        """Verify the Lyapunov function."""
        if not hasattr(lyap, 'verify_positive_definite') or not hasattr(lyap, 'verify_decreasing'):
            return ProofStatus.UNKNOWN
            
        pd_status = lyap.verify_positive_definite()
        
        if pd_status != ProofStatus.VERIFIED:
            return pd_status
            
        return lyap.verify_decreasing(dynamics_fn)


# Demonstration functions

def demonstrate_koopman_lyapunov():
    """Demonstrate Koopman-based Lyapunov learning."""
    print("\n=== Koopman-based Lyapunov Learning ===")
    
    if not HAVE_STABILITY:
        print("Skipping (stability modules not available)")
        # Return placeholder values for minimal implementation
        Q = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        V_koop = QuadraticLyapunov("V_placeholder", Q, domain_ids=["x1", "x2"])
        V_vdp = QuadraticLyapunov("V_vdp_placeholder", Q, domain_ids=["x", "y"])
        return V_koop, V_vdp
    
    # Generate trajectory data for a linear system
    x0 = np.array([1.0, 0.5])
    states, next_states = generate_trajectory_data(linear_system, x0, n_steps=100)
    
    # Create Koopman Lyapunov function
    V_koop = create_koopman_lyapunov(
        states, next_states, 
        dict_type="monomial", 
        dict_params={"degree": 2},
        name="V_linear",
        domain_ids=["x1", "x2"]
    )
    
    # Evaluate at some test points
    test_points = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.5, 0.5])
    ]
    
    print("Evaluating Koopman Lyapunov function at test points:")
    for i, x in enumerate(test_points):
        v = V_koop.evaluate(x)
        x_next = linear_system(x)
        v_next = V_koop.evaluate(x_next)
        print(f"  Point {i+1}: V(x) = {v:.6f}, V(f(x)) = {v_next:.6f}, ratio = {v_next/v:.6f}")
    
    # Try with Van der Pol oscillator
    states_vdp, next_states_vdp = generate_trajectory_data(
        van_der_pol, np.array([1.0, 0.0]), n_steps=200
    )
    
    V_vdp = create_koopman_lyapunov(
        states_vdp, next_states_vdp, 
        dict_type="rbf",
        dict_params={"n_centers": 30, "sigma": 0.5},
        name="V_vdp",
        domain_ids=["x", "y"]
    )
    
    print("\nVan der Pol oscillator with Koopman Lyapunov:")
    x_vdp = np.array([0.5, 0.2])
    v_vdp = V_vdp.evaluate(x_vdp)
    x_next_vdp = van_der_pol(x_vdp)
    v_next_vdp = V_vdp.evaluate(x_next_vdp)
    print(f"  V(x) = {v_vdp:.6f}, V(f(x)) = {v_next_vdp:.6f}, ratio = {v_next_vdp/v_vdp:.6f}")
    
    return V_koop, V_vdp


def demonstrate_incremental_verification(lyap_functions, dynamics_list):
    """Demonstrate incremental verification with dependency tracking."""
    print("\n=== Incremental Verification with Dependency Tracking ===")
    
    if not HAVE_STABILITY:
        print("Skipping (stability modules not available)")
        # Return placeholders
        return None, None
    
    # Create verifier and cache
    verifier = SimpleVerifier()
    cache = ProofCache(cache_dir="./proof_cache")
    
    # Create parallel verifier
    parallel_verifier = ParallelVerifier(
        verifier=verifier,
        cache=cache,
        max_workers=4  # Adjust based on your system
    )
    
    # Create verification tasks
    tasks = []
    for lyap, dynamics in zip(lyap_functions, dynamics_list):
        tasks.append((lyap, dynamics, None))
    
    # Verify batch
    print(f"Verifying {len(tasks)} Lyapunov functions in parallel...")
    results = parallel_verifier.verify_batch(tasks, show_progress=True)
    
    # Print results
    for proof_hash, result in results.items():
        print(f"  {result.lyapunov_name}: {result.status.name} "
             f"(Time: {result.verification_time:.4f}s)")
    
    # Demonstrate dependency tracking
    print("\nDemonstrating dependency tracking:")
    
    # Create a new Lyapunov function that depends on the same concept
    Q2 = np.array([
        [2.0, 0.5],
        [0.5, 1.5]
    ])
    V_new = QuadraticLyapunov("V_new", Q2, domain_ids=["x1", "x2"])
    
    # Add to dependency graph
    cache.dep_graph.add_dependency(
        "V_new_proof", "x1", 
        node_type="proof", parent_type="concept"
    )
    cache.dep_graph.add_dependency(
        "V_new_proof", "x2", 
        node_type="proof", parent_type="concept"
    )
    
    # Simulate a change to concept x1
    old_hashes = {"x1": "hash1", "x2": "hash2"}
    new_hashes = {"x1": "hash1_changed", "x2": "hash2"}
    
    affected = cache.dep_graph.diff_update(old_hashes, new_hashes)
    print(f"  After changing concept x1, {len(affected)} proofs need reverification")
    
    return cache, parallel_verifier


def demonstrate_jit_stability_guard():
    """Demonstrate JIT-compiled stability enforcement."""
    print("\n=== JIT-compiled Stability Enforcement ===")
    
    if not HAVE_STABILITY:
        print("Skipping (stability modules not available)")
        # Return placeholders
        Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        V = QuadraticLyapunov("V_placeholder", Q)
        guard = StabilityGuard(V)
        clf_qp = CLFQuadraticProgramSolver(V, control_dim=1)
        return guard, clf_qp
    
    # Create a quadratic Lyapunov function
    Q = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    V = QuadraticLyapunov("V_quad", Q, domain_ids=["x1", "x2"])
    
    # Define callback for stability violations
    def violation_callback(x_prev, x_now, guard):
        print(f"  Stability violation detected at {x_now}")
        if hasattr(guard, 'delta_V_fn') and hasattr(guard, 'Q'):
            delta = guard.delta_V_fn(x_prev, x_now, guard.Q)
            print(f"  Delta V: {delta:.6f}")
    
    # Create stability guard
    guard = StabilityGuard(
        V,
        threshold=1e-6,
        callback=violation_callback,
        use_jit=HAVE_NUMBA
    )
    
    # Test with a trajectory
    print("Simulating system trajectory...")
    x = np.array([1.0, 0.5])
    trajectory = [x.copy()]
    
    for i in range(5):
        x_prev = x.copy()
        x = linear_system(x)
        trajectory.append(x.copy())
        
        is_stable = guard.step(x_prev, x)
        print(f"  Step {i+1}: stable={is_stable}")
    
    # Test CLF-QP for controlled systems
    print("\nTesting CLF-QP control:")
    
    # Create CLF-QP solver
    clf_qp = CLFQuadraticProgramSolver(
        V,
        control_dim=1,
        gamma=0.1,
        use_jit=HAVE_NUMBA
    )
    
    # Test controlled trajectory
    x = np.array([1.0, 0.5])
    controlled_trajectory = [x.copy()]
    
    for i in range(5):
        # Compute drift and control dynamics
        f_x = linear_system(x)
        g_x = np.array([[1.0], [0.5]])
        
        # Compute control input
        u = clf_qp.step(x, f_x, g_x)
        
        # Update state
        x_prev = x.copy()
        x = controlled_system(x, u)
        controlled_trajectory.append(x.copy())
        
        # Check stability
        is_stable = guard.step(x_prev, x)
        
        print(f"  Step {i+1}: u={u[0]:.4f}, stable={is_stable}")
    
    return guard, clf_qp


def demonstrate_phase_drift_monitoring():
    """Demonstrate phase drift monitoring with adaptive actions."""
    print("\n=== Phase Drift Monitoring & Adaptive Actions ===")
    
    if not HAVE_STABILITY:
        print("Skipping (stability modules not available)")
        # Return placeholder
        concept_map = {
            'HeartbeatPhase': 0,
            'ControllerPhase': 1,
            'NavigationPhase': 2
        }
        monitor = PhaseDriftMonitor(concept_map)
        return monitor
    
    # Create a concept to phase mapping
    concept_map = {
        'HeartbeatPhase': 0,
        'ControllerPhase': 1,
        'NavigationPhase': 2
    }
    
    # Create thresholds
    thresholds = {
        'HeartbeatPhase': math.pi / 8,  # π/8
        'ControllerPhase': math.pi / 4,  # π/4
        'NavigationPhase': math.pi / 6   # π/6
    }
    
    # Create monitor
    monitor = PhaseDriftMonitor(concept_map, thresholds)
    
    # Set reference phases
    for concept in concept_map:
        monitor.set_reference_phase(concept, 0.0)
    
    # Define adaptive action function
    def adaptive_action(concept_id, drift, threshold, action_type):
        print(f"  Action triggered for {concept_id}:")
        print(f"    drift = {drift:.4f}, threshold = {threshold:.4f}")
        
        if action_type == AdaptiveActionType.ADAPT_PLAN:
            print("    Adapting plan based on phase drift")
        elif action_type == AdaptiveActionType.EXECUTE_AGENT:
            print("    Executing agent to correct drift")
            
        return {
            'action': str(action_type),
            'concept': concept_id,
            'drift': drift,
            'status': 'success'
        }
    
    # Register adaptive actions
    monitor.register_adaptive_action(
        concept_id='ControllerPhase',
        threshold=math.pi / 4,
        threshold_type=DriftThresholdType.RADIANS,
        action_type=AdaptiveActionType.ADAPT_PLAN,
        action_fn=adaptive_action,
        description="Adapt plan when ControllerPhase drifts too much"
    )
    
    monitor.register_adaptive_action(
        concept_id='HeartbeatPhase',
        threshold=math.pi / 8,
        threshold_type=DriftThresholdType.RADIANS,
        action_type=AdaptiveActionType.EXECUTE_AGENT,
        action_fn=adaptive_action,
        description="Execute StabilityAgent when HeartbeatPhase drifts"
    )
    
    # Simulate phase drift
    print("Simulating phase drift:")
    
    for i in range(5):
        # Generate random phase values
        phases = {
            'HeartbeatPhase': (i * 0.1) * math.pi / 6,
            'ControllerPhase': (i * 0.2) * math.pi / 4,
            'NavigationPhase': (i * 0.05) * math.pi / 3
        }
        
        print(f"\n  Timestep {i+1}:")
        
        # Measure drift and check for actions
        for concept, phase in phases.items():
            drift = monitor.measure_drift(concept, phase)
            print(f"    {concept}: phase = {phase:.4f}, drift = {drift:.4f}")
        
        # Check for triggered actions
        actions = monitor.check_and_trigger_actions()
        if actions:
            print(f"    {len(actions)} actions triggered")
    
    # Create a Lyapunov predicate for the concepts
    lyap = monitor.create_lyapunov_predicate(list(concept_map.keys()))
    print("\n  Lyapunov predicate created:")
    print(f"    {lyap['symbolic_form']}")
    
    return monitor


def plot_lyapunov_level_sets(lyap_functions, bounds=(-3, 3), grid_size=50):
    """Plot Lyapunov function level sets."""
    print("\n=== Visualizing Lyapunov Functions ===")
    
    if not HAVE_STABILITY:
        print("Skipping (stability modules not available)")
        return
    
    n_funcs = len(lyap_functions)
    fig, axes = plt.subplots(1, n_funcs, figsize=(5*n_funcs, 5))
    
    if n_funcs == 1:
        axes = [axes]
    
    # Create grid
    x = np.linspace(bounds[0], bounds[1], grid_size)
    y = np.linspace(bounds[0], bounds[1], grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Custom colormap: blue for negative, white for zero, red for positive
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("lyapunov", colors, N=100)
    
    for i, (lyap, ax) in enumerate(zip(lyap_functions, axes)):
        # Evaluate Lyapunov function on grid
        Z = np.zeros_like(X)
        for ii in range(grid_size):
            for jj in range(grid_size):
                state = np.array([X[ii, jj], Y[ii, jj]])
                Z[ii, jj] = lyap.evaluate(state)
        
        # Plot level sets
        contour = ax.contourf(X, Y, Z, 20, cmap=cmap)
        fig.colorbar(contour, ax=ax, label="V(x)")
        
        # Plot zero level set
        ax.contour(X, Y, Z, levels=[0], colors='k', linestyles='dashed')
        
        # Add vector field for dynamics
        U, V = np.zeros_like(X), np.zeros_like(Y)
        for ii in range(0, grid_size, 5):
            for jj in range(0, grid_size, 5):
                state = np.array([X[ii, jj], Y[ii, jj]])
                if i == 0:
                    dstate = linear_system(state)
                else:
                    dstate = van_der_pol(state)
                U[ii, jj] = dstate[0]
                V[ii, jj] = dstate[1]
        
        ax.quiver(X[::5, ::5], Y[::5, ::5], U[::5, ::5], V[::5, ::5], 
                 color='white', alpha=0.6)
        
        # Set labels
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"Lyapunov Function: {lyap.name}")
    
    plt.tight_layout()
    plt.savefig("lyapunov_level_sets.png")
    print("Level sets plotted and saved to lyapunov_level_sets.png")


def simulate_integrated_system(lyap, monitor, guard, clf_qp=None):
    """Simulate an integrated system with all stability components."""
    print("\n=== Integrated Stability System Simulation ===")
    
    if not HAVE_STABILITY:
        print("Skipping (stability modules not available)")
        return
    
    # Create a system that uses all components
    print("Simulating integrated system with:")
    print("  - Lyapunov function for stability verification")
    print("  - Phase drift monitoring for concept synchronization")
    print("  - JIT stability guard for runtime enforcement")
    if clf_qp:
        print("  - CLF-QP for stability-enforcing control")
    
    # Initial state and concept phases
    x = np.array([1.0, 0.5])
    concept_phases = {
        'HeartbeatPhase': 0.0,
        'ControllerPhase': 0.0,
        'NavigationPhase': 0.0
    }
    
    # Simulation loop
    print("\nRunning simulation:")
    for i in range(10):
        print(f"\n  Step {i+1}:")
        
        # 1. Check stability with guard
        if i > 0:
            is_stable = guard.step(x_prev, x)
            print(f"    System stability: {is_stable}")
        
        # 2. Update concept phases based on state
        # In a real system, this would come from the oscillator dynamics
        for concept in concept_phases:
            phase_factor = 0.1 if concept == 'HeartbeatPhase' else 0.2
            concept_phases[concept] += phase_factor * (x[0]**2 + x[1]**2)
            concept_phases[concept] = concept_phases[concept] % (2 * math.pi)
        
        # 3. Check phase drift
        for concept, phase in concept_phases.items():
            drift = monitor.measure_drift(concept, phase)
            if drift > 0.1:  # Only report significant drift
                print(f"    {concept}: phase = {phase:.4f}, drift = {drift:.4f}")
        
        # 4. Check for adaptive actions
        actions = monitor.check_and_trigger_actions()
        if actions:
            print(f"    {len(actions)} adaptive actions triggered")
            
            # When action is triggered, modify the dynamics
            if i % 3 == 0:
                print("    Adapting system dynamics based on triggered action")
        
        # 5. Compute control if using CLF-QP
        if clf_qp:
            f_x = linear_system(x)
            g_x = np.array([[1.0], [0.5]])
            u = clf_qp.step(x, f_x, g_x)
            print(f"    Control input: u = {u[0]:.4f}")
            
            # Update state with control
            x_prev = x.copy()
            x = controlled_system(x, u)
        else:
            # Update state with passive dynamics
            x_prev = x.copy()
            x = linear_system(x)
        
        print(f"    State: x = {x}")
    
    print("\nSimulation complete")


def main():
    """Main function demonstrating all components."""
    print("ELFIN DSL with Deep ψ-Sync Integration and Stability Verification")
    print("=" * 70)
    
    # 1. Demonstrate Koopman-based Lyapunov learning
    V_koop, V_vdp = demonstrate_koopman_lyapunov()
    
    # 2. Create a manual quadratic Lyapunov function
    Q = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    V_quad = QuadraticLyapunov("V_quad", Q, domain_ids=["x1", "x2"])
    
    if not HAVE_STABILITY:
        print("\nFull demonstration requires stability modules.")
        print("Run with complete implementation for all features.")
        return
    
    # 3. Demonstrate incremental verification
    lyap_functions = [V_quad, V_koop, V_vdp]
    dynamics_list = [linear_system, linear_system, van_der_pol]
    cache, parallel_verifier = demonstrate_incremental_verification(
        lyap_functions, dynamics_list
    )
    
    # 4. Demonstrate JIT stability guard
    guard, clf_qp = demonstrate_jit_stability_guard()
    
    # 5. Demonstrate phase drift monitoring
    monitor = demonstrate_phase_drift_monitoring()
    
    # 6. Plot Lyapunov level sets
    plot_lyapunov_level_sets([V_quad, V_vdp])
    
    # 7. Simulate integrated system
    simulate_integrated_system(V_quad, monitor, guard, clf_qp)
    
    print("\nDemonstration complete")


if __name__ == "__main__":
    main()
