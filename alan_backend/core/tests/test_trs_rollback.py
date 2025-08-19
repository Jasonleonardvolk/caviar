"""
Tests for TRS (Time-Reversal Symmetry) rollback functionality.

This verifies that our TRS-ODE implementation correctly handles time reversal,
which is critical for the "proof before speak" property of ALAN.
"""

import numpy as np
import pytest
from ..controller.trs_ode import TRSController, State, TRSConfig, VerletIntegrator
from ..controller.trs_ode import HarmonicOscillator, DuffingOscillator
from ..oscillator.banksy_oscillator import BanksyOscillator, BanksyConfig, SpinVector


def test_trs_basic_rollback():
    """Test basic TRS rollback on a simple harmonic oscillator."""
    # Create vector field and controller
    vector_field = HarmonicOscillator(k=1.0)
    config = TRSConfig(dt=0.01, train_steps=100)
    controller = TRSController(
        state_dim=2,
        vector_field=vector_field,
        config=config,
    )
    
    # Initial state
    initial_state = State(np.array([0.5, -0.3]), np.array([0.1, 0.2]))
    
    # Forward integration
    forward_state = controller.forward_integrate(initial_state)
    
    # Backward integration (should return to initial state)
    recovered_state = controller.reverse_integrate(forward_state)
    
    # Calculate TRS loss
    trs_loss = controller.compute_trs_loss(initial_state, forward_state, recovered_state)
    
    # Check that loss is very small (reversible dynamics)
    assert trs_loss < 1e-8, f"TRS loss too large: {trs_loss}"
    
    # Check specific state components
    h_error = np.max(np.abs(recovered_state.h - initial_state.h))
    p_error = np.max(np.abs(recovered_state.p - initial_state.p))
    
    assert h_error < 1e-8, f"Position error too large: {h_error}"
    assert p_error < 1e-8, f"Momentum error too large: {p_error}"


def test_mass_matrix():
    """Test TRS with non-trivial mass matrix."""
    # Create mass matrix (diagonal for simplicity)
    dim = 3
    mass_matrix = np.diag([2.0, 1.0, 3.0])  # Different masses for each dimension
    inv_mass = np.diag([1/2.0, 1.0, 1/3.0])  # Inverse mass matrix
    
    # Create vector field and integrator
    vector_field = HarmonicOscillator(k=1.0)
    integrator = VerletIntegrator(inv_mass=inv_mass)
    
    # Create controller with custom integrator
    config = TRSConfig(dt=0.01, train_steps=100)
    controller = TRSController(
        state_dim=dim,
        vector_field=vector_field,
        config=config,
        integrator=integrator,
    )
    
    # Initial state
    initial_state = State(np.random.randn(dim), np.random.randn(dim))
    
    # Forward and backward integration
    forward_state = controller.forward_integrate(initial_state)
    recovered_state = controller.reverse_integrate(forward_state)
    
    # Calculate TRS loss
    trs_loss = controller.compute_trs_loss(initial_state, forward_state, recovered_state)
    
    # With mass matrix, the loss might be slightly larger but still small
    assert trs_loss < 1e-6, f"TRS loss too large with mass matrix: {trs_loss}"


def test_direction_flag():
    """Test integration direction flag for self-audit."""
    # Create vector field
    vector_field = DuffingOscillator(a=1.0, b=0.3, delta=0.0)  # No damping for reversibility
    
    # Create integrator with direction control
    integrator = VerletIntegrator(direction=1)  # Start with forward
    
    # Create controller with custom integrator
    config = TRSConfig(dt=0.01, train_steps=1)  # Just one step
    controller = TRSController(
        state_dim=1,
        vector_field=vector_field,
        config=config,
        integrator=integrator,
    )
    
    # Initial state
    initial_state = State(np.array([0.5]), np.array([0.1]))
    
    # Take one step forward
    forward_state = controller.forward_integrate(initial_state)
    
    # Change direction to backward
    integrator.set_direction(-1)
    
    # Take one step backward from the forward state
    recovered_state = controller.integrator.step(forward_state, vector_field, config.dt)
    
    # This should recover the initial state
    h_error = np.max(np.abs(recovered_state.h - initial_state.h))
    p_error = np.max(np.abs(recovered_state.p - initial_state.p))
    
    assert h_error < 1e-10, f"Position error in direction flag test: {h_error}"
    assert p_error < 1e-10, f"Momentum error in direction flag test: {p_error}"


def test_dimensionless_trs_loss():
    """Test that TRS loss is dimensionless and scale-invariant."""
    # Create vector field and controller
    vector_field = HarmonicOscillator(k=1.0)
    config = TRSConfig(dt=0.01, train_steps=100)
    controller = TRSController(
        state_dim=1,
        vector_field=vector_field,
        config=config,
    )
    
    # Two initial states with different scales
    small_state = State(np.array([0.01]), np.array([0.01]))
    large_state = State(np.array([100.0]), np.array([100.0]))
    
    # Forward and backward integration for both
    small_forward = controller.forward_integrate(small_state)
    small_recovered = controller.reverse_integrate(small_forward)
    
    large_forward = controller.forward_integrate(large_state)
    large_recovered = controller.reverse_integrate(large_forward)
    
    # Calculate TRS losses
    small_loss = controller.compute_trs_loss(small_state, small_forward, small_recovered)
    large_loss = controller.compute_trs_loss(large_state, large_forward, large_recovered)
    
    # Losses should be comparable despite huge difference in scale
    loss_ratio = large_loss / small_loss if small_loss > 0 else 1.0
    
    # The ratio shouldn't be extreme (we expect some variation, but not orders of magnitude)
    assert 0.1 < loss_ratio < 10.0, f"Loss ratio indicates scale dependence: {loss_ratio}"


def test_banksy_trs_rollback():
    """Test TRS rollback when using the full Banksy oscillator system."""
    # Create Banksy oscillator
    n_oscillators = 5
    banksy_config = BanksyConfig(gamma=0.1, epsilon=0.01, eta_damp=0.0)  # No damping for reversibility
    oscillator = BanksyOscillator(n_oscillators, banksy_config)
    
    # Initialize with specific phases and spins for repeatability
    np.random.seed(42)
    phases = np.random.uniform(0, 2*np.pi, n_oscillators)
    spins = [SpinVector(0, 0, 1) for _ in range(n_oscillators)]  # All spins point up
    
    # Set state
    oscillator.phases = phases.copy()
    oscillator.momenta = np.zeros_like(phases)
    oscillator.spins = spins.copy()
    
    # Take a copy of the initial state
    initial_state = {
        'phases': phases.copy(),
        'momenta': np.zeros_like(phases),
        'spins': [SpinVector(s.x, s.y, s.z) for s in spins],
    }
    
    # Forward steps
    n_steps = 10
    for _ in range(n_steps):
        oscillator.step(spin_substeps=1)  # Use 1 substep for simplicity in testing
    
    # Save the forward state
    forward_state = {
        'phases': oscillator.phases.copy(),
        'momenta': oscillator.momenta.copy(),
        'spins': [SpinVector(s.x, s.y, s.z) for s in oscillator.spins],
    }
    
    # Now reverse the momentum and step backward
    oscillator.momenta = -oscillator.momenta
    
    for _ in range(n_steps):
        oscillator.step(spin_substeps=1)
    
    # Final reversed state
    reversed_state = {
        'phases': oscillator.phases.copy(),
        'momenta': oscillator.momenta.copy(),  # Note: this would be negative of initial
        'spins': [SpinVector(s.x, s.y, s.z) for s in oscillator.spins],
    }
    
    # Check phase recovery (allowing for 2π wrapping)
    for i in range(n_oscillators):
        phase_diff = abs(reversed_state['phases'][i] - initial_state['phases'][i]) % (2 * np.pi)
        phase_diff = min(phase_diff, 2 * np.pi - phase_diff)
        assert phase_diff < 1e-5, f"Phase not recovered for oscillator {i}"
    
    # Since we negated momentum for reverse, we expect final momentum to be negative of initial
    # But initial was zero, so final should be close to zero too
    assert np.max(np.abs(reversed_state['momenta'])) < 1e-5, "Momentum not properly reversed"
    
    # Check spin recovery (should be close to initial since we kept spins simple)
    for i in range(n_oscillators):
        spin_diff = 1.0 - reversed_state['spins'][i].dot(initial_state['spins'][i])
        assert spin_diff < 1e-5, f"Spin not recovered for oscillator {i}"


def test_trs_system_stability():
    """Test system stability over long rollouts with realistic parameters."""
    # Create vector field with some damping
    vector_field = DuffingOscillator(a=1.0, b=0.3, delta=0.01)
    
    # Create controller with realistic steps
    config = TRSConfig(dt=0.01, train_steps=1000)
    controller = TRSController(
        state_dim=2,
        vector_field=vector_field,
        config=config,
    )
    
    # Initial state
    initial_state = State(np.array([0.5, -0.3]), np.array([0.1, 0.2]))
    
    try:
        # This should run without errors or instabilities
        forward_state = controller.forward_integrate(initial_state)
        recovered_state = controller.reverse_integrate(forward_state)
        
        # Even with damping, we expect reasonable recovery
        trs_loss = controller.compute_trs_loss(initial_state, forward_state, recovered_state)
        
        # Loss will be higher with damping, but shouldn't explode
        assert trs_loss < 1.0, f"TRS loss too high in long rollout: {trs_loss}"
        
    except (ValueError, np.linalg.LinAlgError, OverflowError) as e:
        pytest.fail(f"Integration stability test failed with error: {e}")


if __name__ == "__main__":
    # Run the tests directly
    test_trs_basic_rollback()
    test_mass_matrix()
    test_direction_flag()
    test_dimensionless_trs_loss()
    test_banksy_trs_rollback()
    test_trs_system_stability()
    
    print("All tests passed!")
