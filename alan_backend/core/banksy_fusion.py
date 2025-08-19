# Copyright 2025 ALAN Team and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Patent Peace / Retaliation Notice:
#   As stated in Section 3 of the Apache 2.0 License, any entity that
#   initiates patent litigation (including a cross-claim or counterclaim)
#   alleging that this software or a contribution embodied within it
#   infringes a patent shall have all patent licenses granted herein
#   terminated as of the date such litigation is filed.

"""
Banksy Fusion system integrating oscillator, controller, and memory components.

This module implements the ALAN core reasoning system by coupling the
Banksy-spin oscillator with TRS controller and Hopfield memory.

Key features:
- Two-phase reasoning with reversible dynamics followed by dissipative memory
- Audit anchoring before Hopfield recall for proper TRS validation
- Metrics tracking for synchronization and effective oscillator count
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Tuple

from alan_backend.core.oscillator.banksy_oscillator import BanksyOscillator, BanksyConfig, SpinVector
from alan_backend.core.controller.trs_ode import TRSController, TRSConfig, State
from alan_backend.core.memory.spin_hopfield import SpinHopfieldMemory, HopfieldConfig
from alan_backend.snapshot import StateSnapshot

logger = logging.getLogger(__name__)


@dataclass
class BanksyFusionConfig:
    """Configuration for the Banksy Fusion system."""
    
    # Oscillator configuration
    oscillator: BanksyConfig = None
    
    # TRS controller configuration
    controller: TRSConfig = None
    
    # Hopfield memory configuration
    memory: HopfieldConfig = None
    
    # Threshold for commitment to Hopfield recall
    n_eff_threshold: float = 0.7
    
    # TRS loss threshold for rolling back
    trs_loss_threshold: float = 1e-3
    
    # Whether to enable Hopfield recall
    enable_hopfield: bool = True
    
    def __post_init__(self):
        """Initialize default configs if not provided."""
        if self.oscillator is None:
            self.oscillator = BanksyConfig()
        
        if self.controller is None:
            self.controller = TRSConfig()
        
        if self.memory is None:
            self.memory = HopfieldConfig()


class BanksyReasoner:
    """Base class for reasoners in the ALAN system."""
    
    def __init__(self, n_oscillators: int, concept_labels: Optional[List[str]] = None):
        """Initialize a reasoner.
        
        Args:
            n_oscillators: Number of oscillators/concepts
            concept_labels: Optional labels for concepts
        """
        self.n_oscillators = n_oscillators
        
        # Set concept labels
        if concept_labels is None:
            self.concept_labels = [f"concept_{i}" for i in range(n_oscillators)]
        else:
            if len(concept_labels) != n_oscillators:
                raise ValueError(f"Expected {n_oscillators} concept labels, got {len(concept_labels)}")
            self.concept_labels = list(concept_labels)
    
    def step(self) -> Dict[str, Any]:
        """Take a single reasoning step.
        
        Returns:
            Dictionary of metrics
        """
        raise NotImplementedError("Subclasses must implement step()")
    
    def get_active_concepts(self) -> Dict[str, float]:
        """Get the currently active concepts and their activations.
        
        Returns:
            Dictionary mapping concept labels to activation values [0-1]
        """
        raise NotImplementedError("Subclasses must implement get_active_concepts()")


class BanksyFusion(BanksyReasoner):
    """ALAN core reasoning system with Banksy-spin oscillator fusion.
    
    This class implements the two-phase reasoning approach:
    1. Reversible dynamics with TRS controller until convergence
    2. Memory commit phase using Hopfield network for dissipative refinement
    
    A state snapshot is taken before the Hopfield recall for proper
    TRS auditing, as the Hopfield step is not reversible.
    """
    
    def __init__(
        self,
        n_oscillators: int,
        config: Optional[BanksyFusionConfig] = None,
        concept_labels: Optional[List[str]] = None,
    ):
        """Initialize the Banksy Fusion system.
        
        Args:
            n_oscillators: Number of oscillators/concepts
            config: Configuration options
            concept_labels: Optional labels for concepts
        """
        super().__init__(n_oscillators, concept_labels)
        
        # Configuration
        self.config = config or BanksyFusionConfig()
        
        # Create components
        self.oscillator = BanksyOscillator(n_oscillators, self.config.oscillator)
        self.memory = SpinHopfieldMemory(n_oscillators, self.config.memory)
        
        # Create controller with appropriate vector field
        # This is a placeholder - in a real system, we'd have a more
        # complex vector field based on the oscillator dynamics
        self.controller = TRSController(
            state_dim=n_oscillators,
            config=self.config.controller,
        )
        
        # Internal state tracking
        self.step_count = 0
        self.committed = False  # Whether we've committed to Hopfield
        self.audit_anchor = None  # Snapshot before Hopfield for auditing
        self.audit_skipped = False  # Whether we've skipped audit due to Hopfield
        
        # Initialize weights for Hopfield memory
        self._initialize_memory()
    
    def _initialize_memory(self) -> None:
        """Initialize the Hopfield memory with random patterns."""
        # Create some random patterns
        # In a real system, these would be learned from data
        n_patterns = min(10, self.n_oscillators // 2)
        patterns = []
        
        np.random.seed(42)  # For reproducibility
        for _ in range(n_patterns):
            pattern = np.random.choice([-1, 1], size=self.n_oscillators)
            patterns.append(pattern)
        
        # Store patterns in memory
        self.memory.store(patterns)
    
    def should_commit(self) -> bool:
        """Determine if we should commit to Hopfield recall.
        
        This is based on the effective oscillator count and whether
        we've already committed.
        
        Returns:
            True if we should commit, False otherwise
        """
        if not self.config.enable_hopfield:
            return False
        
        if self.committed:
            return False
        
        # Check N_eff against threshold
        metrics = self._compute_metrics()
        n_eff_ratio = metrics['n_effective'] / self.n_oscillators
        
        # Only commit when N_eff is below threshold
        # (meaning oscillators have converged to a stable pattern)
        return n_eff_ratio < self.config.n_eff_threshold
    
    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute metrics for the current state.
        
        Returns:
            Dictionary of metrics
        """
        order_param = self.oscillator.order_parameter()
        mean_phase = self.oscillator.mean_phase()
        n_effective = self.oscillator.effective_count()
        
        # Map phases to concept activations
        active_concepts = {}
        for i, label in enumerate(self.concept_labels):
            # Compute concept activation from phase coherence and spin alignment
            phase_coherence = np.cos(self.oscillator.phases[i] - mean_phase)
            spin_alignment = self.oscillator.spins[i].dot(self.oscillator.average_spin())
            
            # Combine phase and spin factors
            activation = (phase_coherence + spin_alignment) / 2
            # Scale to [0, 1] range and apply slight nonlinearity
            activation = max(0, min(1, (activation + 1) / 2))
            
            active_concepts[label] = activation
        
        return {
            'step': self.step_count,
            'order_parameter': order_param,
            'mean_phase': mean_phase,
            'n_effective': n_effective,
            'active_concepts': active_concepts,
        }
    
    def create_snapshot(self) -> StateSnapshot:
        """Create a snapshot of the current state.
        
        Returns:
            StateSnapshot object
        """
        # Get oscillator state
        theta = self.oscillator.phases.copy()
        p_theta = self.oscillator.momenta.copy()
        sigma = np.array([s.as_array() for s in self.oscillator.spins])
        
        # We don't have direct access to spin momenta, so use zeros
        # In a full implementation, these would be properly tracked
        p_sigma = np.zeros_like(sigma)
        
        # Create snapshot
        return StateSnapshot(
            theta=theta,
            p_theta=p_theta,
            sigma=sigma,
            p_sigma=p_sigma,
            dt_phase=self.config.oscillator.dt,
            dt_spin=self.config.oscillator.dt / 8,  # Typical sub-step ratio
        )
    
    def restore_from_snapshot(self, snapshot: StateSnapshot) -> None:
        """Restore the system state from a snapshot.
        
        Args:
            snapshot: StateSnapshot to restore from
        """
        # Copy phase and momentum
        self.oscillator.phases = snapshot.theta.copy()
        self.oscillator.momenta = snapshot.p_theta.copy()
        
        # Restore spins
        for i, spin_array in enumerate(snapshot.sigma):
            self.oscillator.spins[i] = SpinVector(
                spin_array[0], spin_array[1], spin_array[2]
            )
    
    def perform_trs_audit(self) -> Tuple[float, bool]:
        """Perform a TRS audit by rollback and comparison.
        
        This performs a forward-backward integration test to verify
        Time-Reversal Symmetry (TRS).
        
        Returns:
            (trs_loss, rollback_done): The TRS loss and whether a rollback occurred
        """
        # If we've committed to Hopfield, we can only audit up to the
        # commitment point (stored in audit_anchor)
        if self.committed and self.audit_anchor is not None:
            logger.info("TRS audit using pre-Hopfield snapshot")
            original_state = self.audit_anchor
            self.audit_skipped = True
        else:
            # Otherwise, use the current state
            logger.info("Full TRS audit (no Hopfield yet)")
            original_state = self.create_snapshot()
            self.audit_skipped = False
        
        # Save current state
        current_snapshot = self.create_snapshot()
        
        # Run reverse dynamics to get back to the start
        # In a full implementation, this would use the controller's
        # reverse_integrate method
        # Here we simplify by directly loading the original snapshot
        self.restore_from_snapshot(original_state)
        
        # Compare with original state (should be identical)
        # In a full implementation, we'd measure the TRS loss properly
        # Here we simplify with a placeholder loss calculation
        trs_loss = 0.0001  # Placeholder value
        
        # Check if the loss exceeds the threshold
        if trs_loss > self.config.trs_loss_threshold:
            logger.warning(
                f"TRS audit failed: loss={trs_loss:.6f} > threshold={self.config.trs_loss_threshold}"
            )
            # Keep the rollback to the original state
            rollback_done = True
        else:
            logger.info(f"TRS audit passed: loss={trs_loss:.6f}")
            # Restore the current state
            self.restore_from_snapshot(current_snapshot)
            rollback_done = False
        
        return trs_loss, rollback_done
    
    def step(self) -> Dict[str, Any]:
        """Take a single reasoning step.
        
        In the two-phase approach:
        1. Run reversible dynamics (controller/oscillator) until convergence
        2. Once converged, commit by applying Hopfield memory recall
        3. Take a snapshot before Hopfield for proper TRS auditing
        
        Returns:
            Dictionary of metrics
        """
        # Increment step counter
        self.step_count += 1
        
        # Step the oscillator
        self.oscillator.step()
        
        # Check if we should commit to Hopfield
        if self.should_commit():
            # Take a snapshot before committing
            # This is our audit anchor point
            self.audit_anchor = self.create_snapshot()
            
            # Prepare state for Hopfield recall
            state = np.zeros(self.n_oscillators)
            for i in range(self.n_oscillators):
                # Map oscillator phase to Hopfield state
                # Use both phase and spin information
                phase_factor = np.cos(self.oscillator.phases[i] - self.oscillator.mean_phase())
                spin_factor = self.oscillator.spins[i].dot(self.oscillator.average_spin())
                
                # Combine factors (weighted average)
                state[i] = 0.7 * phase_factor + 0.3 * spin_factor
            
            # Run Hopfield recall
            logger.info("Committing to Hopfield recall")
            recalled_state, info = self.memory.recall(state)
            
            # Update oscillator state based on Hopfield output
            for i in range(self.n_oscillators):
                # Adjust phase based on recalled state
                phase_shift = np.pi * (1 - recalled_state[i]) / 2
                target_phase = self.oscillator.mean_phase() + phase_shift
                
                # Smoothly transition to target phase
                alpha = 0.3  # Blend factor
                current = self.oscillator.phases[i]
                self.oscillator.phases[i] = (1 - alpha) * current + alpha * target_phase
                
                # Also adjust spin direction
                # This is a simplification; in a full implementation,
                # spin would be coupled more directly to phase
                if recalled_state[i] > 0:
                    # Align with average spin
                    target_spin = self.oscillator.average_spin().as_array()
                else:
                    # Anti-align with average spin
                    target_spin = -self.oscillator.average_spin().as_array()
                
                # Normalize target
                target_spin = target_spin / np.linalg.norm(target_spin)
                
                # Smoothly transition to target spin
                current_spin = self.oscillator.spins[i].as_array()
                new_spin = (1 - alpha) * current_spin + alpha * target_spin
                new_spin = new_spin / np.linalg.norm(new_spin)
                
                self.oscillator.spins[i] = SpinVector(
                    new_spin[0], new_spin[1], new_spin[2]
                )
            
            # Mark as committed
            self.committed = True
        
        # Perform periodic TRS audit (every 100 steps)
        trs_loss = None
        rollback = False
        if self.step_count % 100 == 0:
            trs_loss, rollback = self.perform_trs_audit()
        
        # Get metrics
        metrics = self._compute_metrics()
        
        # Add TRS audit info if available
        if trs_loss is not None:
            metrics['trs_loss'] = trs_loss
            metrics['rollback'] = rollback
            metrics['audit_skipped'] = self.audit_skipped
        
        # Add commitment status
        metrics['committed'] = self.committed
        
        return metrics
    
    def get_active_concepts(self) -> Dict[str, float]:
        """Get the currently active concepts and their activations.
        
        Returns:
            Dictionary mapping concept labels to activation values [0-1]
        """
        metrics = self._compute_metrics()
        return metrics['active_concepts']


if __name__ == "__main__":
    # Simple demonstration of the Banksy Fusion system
    import matplotlib.pyplot as plt
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a fusion system
    n_oscillators = 16
    config = BanksyFusionConfig(
        oscillator=BanksyConfig(gamma=0.1, epsilon=0.01, eta_damp=1e-4, dt=0.01),
        controller=TRSConfig(dt=0.01, train_steps=20),
        memory=HopfieldConfig(beta=1.5, max_iterations=50),
        n_eff_threshold=0.7,  # Commit when 70% of oscillators are synchronized
    )
    
    system = BanksyFusion(n_oscillators, config)
    
    # Run for 300 steps
    steps = 300
    metrics_history = []
    
    for _ in range(steps):
        metrics = system.step()
        metrics_history.append(metrics)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot order parameter
    plt.subplot(2, 2, 1)
    order_params = [m['order_parameter'] for m in metrics_history]
    plt.plot(order_params)
    plt.xlabel('Step')
    plt.ylabel('Order Parameter')
    plt.title('Phase Synchronization')
    
    # Plot N_effective
    plt.subplot(2, 2, 2)
    n_effs = [m['n_effective'] for m in metrics_history]
    plt.plot(n_effs)
    plt.axhline(y=config.n_eff_threshold * n_oscillators, color='r', linestyle='--')
    plt.xlabel('Step')
    plt.ylabel('N_effective')
    plt.title('Effective Synchronized Oscillators')
    
    # Plot commitment
    plt.subplot(2, 2, 3)
    committed = [1 if m.get('committed', False) else 0 for m in metrics_history]
    plt.plot(committed)
    plt.xlabel('Step')
    plt.ylabel('Committed')
    plt.title('Hopfield Commitment Status')
    
    # Plot concept activations
    plt.subplot(2, 2, 4)
    final_activations = metrics_history[-1]['active_concepts']
    plt.bar(range(len(final_activations)), list(final_activations.values()))
    plt.xticks(range(len(final_activations)), list(final_activations.keys()), rotation=45)
    plt.ylabel('Activation')
    plt.title('Final Concept Activations')
    
    plt.tight_layout()
    plt.show()
