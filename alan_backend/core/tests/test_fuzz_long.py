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
Long-horizon fuzzing tests for ALAN system.

These tests run the system for long periods (many iterations)
to ensure stability and proper time-reversal symmetry in extreme cases.
"""

import pytest
import numpy as np
from hypothesis import given, settings, strategies as st

from alan_backend.core.oscillator.banksy_oscillator import BanksyOscillator, BanksyConfig, SpinVector
from alan_backend.core.controller.trs_ode import TRSController, TRSConfig, State, VerletIntegrator, HarmonicOscillator
from alan_backend.core.memory.spin_hopfield import SpinHopfieldMemory, HopfieldConfig
from alan_backend.core.banksy_fusion import BanksyFusion, BanksyFusionConfig


class TestLongHorizon:
    """Long-horizon stability tests for the core components."""
    
    def test_oscillator_long_stability(self):
        """Test that oscillator remains stable over extremely long runs."""
        # Create oscillator with default parameters
        n_oscillators = 16
        config = BanksyConfig(gamma=0.1, epsilon=0.01, eta_damp=1e-4, dt=0.01)
        oscillator = BanksyOscillator(n_oscillators, config)
        
        # Long run (100,000 steps)
        n_steps = 100_000
        
        # Save initial state
        initial_phases = oscillator.phases.copy()
        initial_momenta = oscillator.momenta.copy()
        initial_spins = [SpinVector(s.x, s.y, s.z) for s in oscillator.spins]
        
        # Initial energy (approximated by order parameter deviation from 1)
        initial_order = oscillator.order_parameter()
        
        # Run for many steps
        for i in range(n_steps):
            oscillator.step()
            
            # Periodically check for instability (NaN values)
            if i % 10_000 == 0:
                # Check for NaN
                assert not np.isnan(oscillator.phases).any(), f"NaN detected in phases at step {i}"
                assert not np.isnan(oscillator.momenta).any(), f"NaN detected in momenta at step {i}"
                for s in oscillator.spins:
                    assert not np.isnan(s.as_array()).any(), f"NaN detected in spins at step {i}"
        
        # Check final state metrics (should be reasonable)
        final_order = oscillator.order_parameter()
        # Order parameter shouldn't decay significantly over time
        assert abs(final_order - initial_order) < 0.5, "Order parameter degraded significantly"


    def test_trs_controller_round_trip(self):
        """Test TRS controller on a very long round-trip integration."""
        # Create a simple harmonic oscillator
        vector_field = HarmonicOscillator(k=0.5)
        
        # Create a TRS controller with high precision integrator
        config = TRSConfig(dt=0.01, train_steps=10_000)
        controller = TRSController(
            state_dim=4,
            vector_field=vector_field,
            config=config,
        )
        
        # Create initial state
        np.random.seed(42)  # Fixed seed for reproducibility
        initial_state = State(np.random.normal(0, 1, 4), np.random.normal(0, 1, 4))
        
        # Forward integrate for 10,000 steps
        forward_state = controller.forward_integrate(initial_state)
        
        # Backward integrate for 10,000 steps
        reversed_state = controller.reverse_integrate(forward_state)
        
        # Calculate TRS loss
        trs_loss = controller.compute_trs_loss(initial_state, forward_state, reversed_state)
        
        # Loss should remain small even for long round trips
        assert trs_loss < 1e-3, f"TRS loss too large: {trs_loss}"


    def test_hopfield_energy_monotonicity(self):
        """Test that Hopfield memory maintains energy monotonicity for both binary and continuous states."""
        # Test with binary states
        self._test_hopfield_mode(binary=True)
        
        # Test with continuous states
        self._test_hopfield_mode(binary=False)
    
    def _test_hopfield_mode(self, binary: bool):
        """Helper to test Hopfield memory in binary or continuous mode."""
        size = 100
        config = HopfieldConfig(
            beta=1.0,
            binary=binary,
            max_iterations=1000,
            energy_threshold=1e-9,
        )
        memory = SpinHopfieldMemory(size, config)
        
        # Create random weight matrix
        np.random.seed(42)
        weights = np.random.normal(0, 0.1, (size, size))
        weights = (weights + weights.T) / 2  # Make symmetric
        np.fill_diagonal(weights, 0)  # No self-connections
        memory.set_weights(weights)
        
        # Create random initial state
        if binary:
            initial_state = np.random.choice([-1, 1], size=size)
        else:
            initial_state = np.random.uniform(-1, 1, size=size)
        
        # Run recall
        state, info = memory.recall(initial_state)
        
        # Check energy history is monotonically decreasing
        energy_history = info['energy_history']
        assert len(energy_history) > 2, "Energy history too short"
        
        # Each energy value should be less than or equal to the previous one
        for i in range(1, len(energy_history)):
            assert energy_history[i] <= energy_history[i-1], \
                f"Energy increased at step {i-1}→{i}: {energy_history[i-1]} → {energy_history[i]}"


@settings(deadline=None, max_examples=15)
@given(st.integers(min_value=8, max_value=32), st.booleans())
def test_fusion_system_long_run(n_oscillators, use_binary_mode):
    """Test the complete ALAN system for very long runs generated by Hypothesis."""
    # Configure system components
    oscillator_config = BanksyConfig(gamma=0.1, epsilon=0.01, eta_damp=1e-4, dt=0.01)
    trs_config = TRSConfig(dt=0.01, train_steps=10)
    memory_config = HopfieldConfig(beta=1.5, binary=use_binary_mode, max_iterations=50)
    
    # Create fusion config
    fusion_config = BanksyFusionConfig(
        oscillator=oscillator_config,
        controller=trs_config,
        memory=memory_config,
    )
    
    # Create fusion system
    system = BanksyFusion(n_oscillators, fusion_config)
    
    # Run for a moderate number of steps (250)
    # Note: For extremely long runs this would be 1,000,000, but that's too slow for CI
    steps = 250
    
    # Take initial snapshot for rollback test
    snapshot = system._compute_metrics()
    
    # Run forward
    for _ in range(steps):
        system.step()
    
    metrics = system._compute_metrics()
    
    # Check that N_effective didn't collapse to 0
    assert metrics['n_effective'] > 0, "Effective oscillator count should not be zero"
    
    # Check the mean phase is not NaN
    assert not np.isnan(metrics['mean_phase']), "Mean phase is NaN"
    
    # Check order parameter didn't decay to zero
    assert metrics['order_parameter'] > 0.1, "Order parameter should not decay to zero"


@settings(deadline=None, max_examples=5)
@given(st.integers(min_value=4, max_value=16))
def test_trs_stability_property(n_dim):
    """Property-based test for TRS controller stability."""
    # Create vector field with random parameters
    np.random.seed(42)
    k = np.random.uniform(0.5, 2.0)
    vector_field = HarmonicOscillator(k=k)
    
    # Create controller
    config = TRSConfig(dt=0.01, train_steps=1000)
    controller = TRSController(
        state_dim=n_dim,
        vector_field=vector_field,
        config=config,
    )
    
    # Random initial state
    h = np.random.normal(0, 1, n_dim)
    p = np.random.normal(0, 1, n_dim)
    initial_state = State(h, p)
    
    # Forward for 1000 steps
    forward_state = controller.forward_integrate(initial_state)
    
    # Reverse to get back to start
    reversed_state = controller.reverse_integrate(forward_state)
    
    # Compute TRS loss
    trs_loss = controller.compute_trs_loss(initial_state, forward_state, reversed_state)
    
    # Loss should be small for a well-implemented system
    assert trs_loss < 1e-2, f"TRS loss too large: {trs_loss}"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
