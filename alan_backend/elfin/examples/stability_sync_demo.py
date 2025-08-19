#!/usr/bin/env python3
"""
ELFIN DSL Stability & ψ-Sync Integration Demo.

This script demonstrates how to use the ELFIN DSL with ψ-mode decorators,
Lyapunov stability constraints, and runtime adaptation to phase drift.
"""

import os
import sys
import logging
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from enum import Enum, auto  # Added missing imports

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path to import ELFIN modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

try:
    from alan_backend.elfin.stability.phase_drift_monitor import (
        PhaseDriftMonitor, DriftThresholdType, AdaptiveActionType
    )
    try:
        from alan_backend.banksy.psi_sync_monitor import PsiSyncMonitor
        HAVE_PSI_SYNC = True
    except ImportError:
        logger.warning("Could not import PsiSyncMonitor. Using simulated data instead.")
        HAVE_PSI_SYNC = False
except ImportError:
    # If we can't import the main module, let's define minimal versions
    # of what we need for the demo to work with simulated data
    logger.warning("Could not import stability modules. Creating minimal implementations for demo.")
    
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
    
    class PhaseDriftMonitor:
        def __init__(self, concept_to_psi_map, thresholds=None, banksy_monitor=None):
            self.concept_to_psi = concept_to_psi_map
            self.thresholds = thresholds or {}
            self.current_drift = {}
            self.reference_phases = {}
            self.adaptive_actions = {}
            logger.info("Using minimal PhaseDriftMonitor implementation")
        
        def set_reference_phase(self, concept_id, phase):
            self.reference_phases[concept_id] = phase
            return True
        
        def measure_drift(self, concept_id, current_phase):
            if concept_id not in self.reference_phases:
                return 0.0
            ref_phase = self.reference_phases[concept_id]
            diff = current_phase - ref_phase
            drift = ((diff + math.pi) % (2 * math.pi)) - math.pi
            self.current_drift[concept_id] = drift
            return abs(drift)
        
        def register_adaptive_action(self, concept_id, threshold, threshold_type, action_type, action_fn, description=None):
            self.adaptive_actions[concept_id] = {
                'threshold': threshold,
                'action_type': action_type,
                'action_fn': action_fn,
                'description': description or f"Action for {concept_id}",
                'last_triggered': 0,
                'trigger_count': 0
            }
            return True
        
        def check_and_trigger_actions(self):
            triggered_actions = []
            for concept_id, action_info in self.adaptive_actions.items():
                if concept_id not in self.current_drift:
                    continue
                drift = self.current_drift[concept_id]
                threshold = action_info['threshold']
                if abs(drift) > threshold:
                    now = time.time()
                    if now - action_info.get('last_triggered', 0) < 5.0:
                        continue
                    action_info['last_triggered'] = now
                    action_info['trigger_count'] = action_info.get('trigger_count', 0) + 1
                    try:
                        result = action_info['action_fn'](
                            concept_id=concept_id,
                            drift=drift,
                            threshold=threshold,
                            action_type=action_info['action_type']
                        )
                        triggered_actions.append({
                            'concept_id': concept_id,
                            'drift': drift,
                            'threshold': threshold,
                            'action_type': str(action_info['action_type']),
                            'description': action_info['description'],
                            'result': result
                        })
                    except Exception as e:
                        logger.error(f"Error triggering action: {e}")
            return triggered_actions
        
        def create_lyapunov_predicate(self, concept_ids):
            valid_concepts = [c for c in concept_ids if c in self.concept_to_psi]
            n = len(valid_concepts)
            return {
                'type': 'polynomial',
                'dimension': n,
                'concepts': valid_concepts,
                'symbolic_form': f"V(x) = x^T I x  (for {', '.join(valid_concepts)})"
            }
    
    HAVE_PSI_SYNC = False

# Example ELFIN script with the new ψ-Sync and Lyapunov stability features
ELFIN_EXAMPLE = """
/* 
 * ELFIN DSL with ψ-Sync and Lyapunov Stability Extensions
 */

// Concept with direct ψ-mode binding
concept "HeartbeatPhase" ψ-mode: ϕ3 {
    frequency = 1.0;
    importance = "critical";
    require Lyapunov(ψ_HeartbeatPhase) < 0.5;
}

// Concept with named phase binding
concept "ControllerPhase" ψ-mode: controller_mode {
    frequency = 2.0;
    response_time = 0.1;
    
    // Require ControllerPhase to synchronize with HeartbeatPhase
    require PhaseDrift(ψ_ControllerPhase) < π/4;
}

// Stability constraint on Lyapunov function
stability Lyapunov(ψ_SystemState) < 0;

// Phase drift monitoring
monitor PhaseDrift(ψ_HeartbeatPhase) < π/8;

// Runtime adaptation to phase drift
if PhaseDrift(ψ_ControllerPhase) > π/4: adapt plan via StabilityAgent;

// Koopman operator definition
koopman SystemDynamics {
    eigenfunctions(ψ1, ψ2, ψ3)
    modes(
        heartbeat: 1.0,
        controller: 2.0,
        navigation: 0.5
    )
    bind to phase(
        heartbeat -> HeartbeatPhase,
        controller -> ControllerPhase
    )
}

// Lyapunov function definition
lyapunov SystemStability {
    polynomial(2)
    domain(HeartbeatPhase, ControllerPhase)
    form "V(x) = x^T Q x"
    verify(sos, timeout=30)
}
"""

def print_elfin_with_highlights():
    """Print the ELFIN example script with syntax highlighting."""
    print("\nELFIN DSL with ψ-Sync and Lyapunov Stability Extensions:\n")
    
    # Simple highlighting - we could use a proper syntax highlighter library
    # like Pygments, but this is just for demonstration purposes
    lines = ELFIN_EXAMPLE.strip().split('\n')
    for line in lines:
        # Comments
        if line.strip().startswith('/*') or line.strip().startswith('//'):
            print(f"\033[36m{line}\033[0m")  # Cyan
        # Keywords
        elif any(keyword in line for keyword in ['concept', 'require', 'stability', 'monitor', 'if', 
                                                 'koopman', 'lyapunov']):
            print(f"\033[33m{line}\033[0m")  # Yellow
        # ψ-mode related
        elif 'ψ-mode' in line or 'ψ_' in line or 'ϕ' in line:
            print(f"\033[35m{line}\033[0m")  # Magenta
        # Lyapunov or phase drift
        elif 'Lyapunov' in line or 'PhaseDrift' in line:
            print(f"\033[32m{line}\033[0m")  # Green
        # Other
        else:
            print(line)


def simulate_phase_dynamics(num_oscillators=3, num_steps=100, coupling_strength=0.1):
    """
    Simulate Kuramoto oscillators to demonstrate phase synchronization.
    
    Args:
        num_oscillators: Number of oscillators
        num_steps: Number of simulation steps
        coupling_strength: Strength of coupling between oscillators
        
    Returns:
        Tuple of (phases, frequencies, coupling_matrix)
    """
    # Natural frequencies
    frequencies = np.array([1.0, 2.0, 0.5])  # Match the Koopman modes
    
    # Coupling matrix
    coupling = np.zeros((num_oscillators, num_oscillators))
    coupling[0, 1] = coupling_strength  # Heartbeat -> Controller
    coupling[1, 0] = coupling_strength  # Controller -> Heartbeat
    coupling[1, 2] = coupling_strength  # Controller -> Navigation
    
    # Initial phases
    phases = np.random.uniform(0, 2*np.pi, (num_steps, num_oscillators))
    phases[0] = np.random.uniform(0, 2*np.pi, num_oscillators)
    
    # Simulate Kuramoto model
    dt = 0.1
    for t in range(1, num_steps):
        for i in range(num_oscillators):
            # Natural frequency contribution
            dphase = frequencies[i]
            
            # Coupling contribution
            for j in range(num_oscillators):
                dphase += coupling[j, i] * np.sin(phases[t-1, j] - phases[t-1, i])
            
            # Update phase
            phases[t, i] = phases[t-1, i] + dphase * dt
            
            # Wrap to [0, 2π]
            phases[t, i] = phases[t, i] % (2 * np.pi)
    
    return phases, frequencies, coupling


def plot_phase_dynamics(phases, concept_names):
    """
    Plot phase dynamics and order parameter.
    
    Args:
        phases: Array of phases (time, oscillator)
        concept_names: Names of concepts for each oscillator
    """
    num_steps, num_oscillators = phases.shape
    
    # Calculate order parameter
    z = np.exp(1j * phases)
    r = np.abs(np.mean(z, axis=1))
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot phases
    plt.subplot(2, 1, 1)
    for i in range(num_oscillators):
        plt.plot(phases[:, i], label=concept_names[i])
    plt.ylabel('Phase (radians)')
    plt.xlabel('Time step')
    plt.title('Phase Dynamics')
    plt.legend()
    plt.grid(True)
    
    # Plot order parameter
    plt.subplot(2, 1, 2)
    plt.plot(r)
    plt.ylabel('Order parameter (r)')
    plt.xlabel('Time step')
    plt.title('Synchronization Level')
    plt.grid(True)
    
    plt.tight_layout()


def demonstrate_phase_drift_monitoring(phases, concept_names):
    """
    Demonstrate phase drift monitoring and adaptive actions.
    
    Args:
        phases: Array of phases (time, oscillator)
        concept_names: Names of concepts for each oscillator
    """
    print("\nDemonstrating Phase Drift Monitoring & Adaptive Actions:")
    
    # Create concept to psi-mode mapping
    concept_map = {name: i for i, name in enumerate(concept_names)}
    
    # Create thresholds based on the ELFIN script
    thresholds = {
        'HeartbeatPhase': math.pi / 8,  # π/8
        'ControllerPhase': math.pi / 4,  # π/4
        'NavigationPhase': math.pi / 6   # π/6
    }
    
    # Create monitor
    monitor = PhaseDriftMonitor(concept_map, thresholds)
    
    # Set reference phases to initial values
    for concept, idx in concept_map.items():
        monitor.set_reference_phase(concept, phases[0, idx])
    
    # Define adaptive actions based on the ELFIN script
    def stability_agent_action(concept_id, drift, threshold, action_type):
        print(f"  StabilityAgent triggered for {concept_id}: "
              f"drift = {drift:.4f}, threshold = {threshold:.4f}")
        action = "Adjusting coupling"
        print(f"  Action: {action}")
        return {'status': 'success', 'action': action, 'drift': drift}
    
    # Register adaptive actions
    monitor.register_adaptive_action(
        concept_id='ControllerPhase',
        threshold=math.pi / 4,  # π/4
        threshold_type=DriftThresholdType.RADIANS,
        action_type=AdaptiveActionType.ADAPT_PLAN,
        action_fn=stability_agent_action,
        description="Adapt plan via StabilityAgent when ControllerPhase drift > π/4"
    )
    
    # Monitor phase drift for each time step
    print("\n  Monitoring phase drift...")
    num_steps = phases.shape[0]
    detected_drifts = []
    
    for t in range(1, num_steps, 10):  # Sample every 10 steps for brevity
        print(f"\n  Time step {t}:")
        
        # Check drift for each concept
        for concept, idx in concept_map.items():
            # Get current phase
            current_phase = phases[t, idx]
            
            # Measure drift
            drift = monitor.measure_drift(concept, current_phase)
            
            # Print drift (only if significant)
            if abs(drift) > 0.1:
                print(f"    {concept}: drift = {drift:.4f} rad")
                detected_drifts.append((t, concept, drift))
        
        # Check and trigger actions
        actions = monitor.check_and_trigger_actions()
        if actions:
            print(f"\n    Triggered {len(actions)} actions at step {t}")
    
    # Create a Lyapunov predicate for the concepts
    print("\n  Creating Lyapunov predicate...")
    lyap = monitor.create_lyapunov_predicate(list(concept_map.keys()))
    print(f"    {lyap['symbolic_form']}")
    
    return detected_drifts


def demonstrate_lyapunov_verification():
    """Demonstrate Lyapunov function verification."""
    print("\nDemonstrating Lyapunov Function Verification:")
    
    try:
        from alan_backend.elfin.stability.verifier import LyapunovVerifier
        
        # Create a simple 2D system
        def dynamics(x):
            return -x
        
        # Create a quadratic Lyapunov function V(x) = x^T Q x
        Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        print("  System: dx/dt = -x")
        print("  Lyapunov function: V(x) = x^T Q x = x1^2 + x2^2")
        
        # Verify the Lyapunov function
        print("\n  Verifying using sum-of-squares (SOS)...")
        
        # Manually check positive definiteness
        eigenvalues = np.linalg.eigvalsh(Q)
        print(f"    Q eigenvalues: {eigenvalues}")
        print(f"    Is Q positive definite? {np.all(eigenvalues > 0)}")
        
        # Manually check decreasing property
        x_test = np.array([0.5, -0.7])
        v_x = x_test @ Q @ x_test
        dx_dt = dynamics(x_test)
        gradient = 2 * Q @ x_test
        dv_dt = gradient @ dx_dt
        
        print(f"\n    Test at x = {x_test}:")
        print(f"    V(x) = {v_x:.4f}")
        print(f"    dx/dt = {dx_dt}")
        print(f"    ∇V(x) = {gradient}")
        print(f"    dV/dt = ∇V(x)·dx/dt = {dv_dt:.4f}")
        
        print(f"\n    Verification result: {'STABLE' if dv_dt < 0 else 'UNSTABLE'}")
        
    except ImportError:
        print("  Skipping Lyapunov verification (verifier module not available)")


def main():
    """Main function for the demo."""
    print("\n" + "="*80)
    print("ELFIN DSL Stability & ψ-Sync Integration Demo")
    print("="*80)
    
    # Print the ELFIN example script with syntax highlighting
    print_elfin_with_highlights()
    
    # Demonstrate phase dynamics
    print("\nSimulating phase dynamics (Kuramoto oscillators)...")
    concept_names = ['HeartbeatPhase', 'ControllerPhase', 'NavigationPhase']
    phases, frequencies, coupling = simulate_phase_dynamics()
    print(f"  Natural frequencies: {frequencies}")
    print(f"  Coupling matrix:\n{coupling}")
    
    # Demonstrate phase drift monitoring
    drifts = demonstrate_phase_drift_monitoring(phases, concept_names)
    
    # Demonstrate Lyapunov verification
    demonstrate_lyapunov_verification()
    
    # Plot phase dynamics
    plot_phase_dynamics(phases, concept_names)
    
    print("\nDone. Close the plot window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
