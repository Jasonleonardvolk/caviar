"""
Unit tests for the TrajectorySampler class.

Tests the functionality of the TrajectorySampler for generating training data
for neural Lyapunov functions.
"""

import pytest
import numpy as np
from ..samplers.trajectory_sampler import TrajectorySampler


# Test systems
def linear_system(x):
    """Linear stable system dx/dt = -0.5*x."""
    return -0.5 * x

def test_trajectory_sampler_initialization():
    """Test initialization of TrajectorySampler."""
    # Valid initialization
    dim = 2
    domain = (np.array([-1, -1]), np.array([1, 1]))
    sampler = TrajectorySampler(linear_system, dim, domain)
    
    assert sampler.dim == dim
    assert np.array_equal(sampler.low, domain[0])
    assert np.array_equal(sampler.high, domain[1])
    assert sampler.batch_size == 1024  # Default value
    
    # Test with custom batch size
    batch_size = 128
    sampler = TrajectorySampler(linear_system, dim, domain, batch_size=batch_size)
    assert sampler.batch_size == batch_size
    
    # Test with invalid inputs
    with pytest.raises(ValueError):
        # Dimension mismatch
        TrajectorySampler(
            linear_system, 
            dim, 
            (np.array([-1]), np.array([1, 1]))
        )
    
    with pytest.raises(ValueError):
        # Lower bound greater than upper bound
        TrajectorySampler(
            linear_system, 
            dim, 
            (np.array([1, 1]), np.array([-1, -1]))
        )

def test_random_batch():
    """Test random batch generation."""
    dim = 3
    domain = (np.array([-1, -2, -3]), np.array([1, 2, 3]))
    batch_size = 100
    
    sampler = TrajectorySampler(linear_system, dim, domain, batch_size=batch_size)
    
    # Generate batch
    x, dx = sampler.random_batch()
    
    # Check shapes
    assert x.shape == (batch_size, dim)
    assert dx.shape == (batch_size, dim)
    
    # Check if samples are within domain bounds
    assert np.all(x >= domain[0])
    assert np.all(x <= domain[1])
    
    # Check if derivatives are correctly computed
    expected_dx = linear_system(x)
    assert np.allclose(dx, expected_dx)

def test_add_counterexamples():
    """Test adding counterexamples to the sampler."""
    dim = 2
    domain = (np.array([-1, -1]), np.array([1, 1]))
    
    sampler = TrajectorySampler(linear_system, dim, domain)
    
    # Initially no counterexamples
    assert sampler.get_counterexample_count() == 0
    
    # Add valid counterexamples
    ce1 = np.array([0.5, 0.5])
    ce2 = np.array([-0.5, 0.5])
    sampler.add_counterexamples([ce1, ce2])
    
    assert sampler.get_counterexample_count() == 2
    
    # Add invalid counterexample (wrong shape)
    ce3 = np.array([0.5, 0.5, 0.5])
    sampler.add_counterexamples([ce3])
    
    # Should still be 2 (invalid one ignored)
    assert sampler.get_counterexample_count() == 2
    
    # Check clearing counterexamples
    sampler.clear_counterexamples()
    assert sampler.get_counterexample_count() == 0

def test_balanced_batch():
    """Test balanced batch generation with counterexamples."""
    dim = 2
    domain = (np.array([-1, -1]), np.array([1, 1]))
    batch_size = 100
    
    sampler = TrajectorySampler(linear_system, dim, domain, batch_size=batch_size)
    
    # No counterexamples -> random batch
    x1, _ = sampler.balanced_batch()
    
    # Add counterexamples
    n_counterexamples = 10
    counterexamples = [
        np.random.uniform(-1, 1, size=dim) for _ in range(n_counterexamples)
    ]
    sampler.add_counterexamples(counterexamples)
    
    # Generate balanced batch
    x2, dx2 = sampler.balanced_batch()
    
    # Check shapes
    assert x2.shape == (batch_size, dim)
    assert dx2.shape == (batch_size, dim)
    
    # Since we have fewer counterexamples than half the batch size,
    # all should be included
    assert sampler.get_counterexample_count() <= batch_size // 2
    
    # Add many counterexamples
    many_counterexamples = [
        np.random.uniform(-1, 1, size=dim) for _ in range(batch_size)
    ]
    sampler.add_counterexamples(many_counterexamples)
    
    # Generate balanced batch
    x3, _ = sampler.balanced_batch()
    
    # Should still have batch_size samples
    assert x3.shape == (batch_size, dim)

def test_simulate_trajectory():
    """Test trajectory simulation."""
    dim = 2
    domain = (np.array([-5, -5]), np.array([5, 5]))
    
    sampler = TrajectorySampler(linear_system, dim, domain)
    
    # Test continuous-time simulation
    x0 = np.array([1.0, 1.0])
    steps = 10
    dt = 0.1
    
    states, derivatives = sampler.simulate_trajectory(x0, steps, dt)
    
    # Check shapes
    assert states.shape == (steps + 1, dim)
    assert derivatives.shape == (steps + 1, dim)
    
    # Check initial state
    assert np.array_equal(states[0], x0)
    
    # For linear system x' = -0.5*x, we can check against analytical solution
    # x(t) = x0 * exp(-0.5*t)
    for i in range(1, steps + 1):
        t = i * dt
        expected_state = x0 * np.exp(-0.5 * t)
        assert np.allclose(states[i], expected_state, rtol=1e-2)
        
    # Test discrete-time simulation
    discrete_sampler = TrajectorySampler(
        lambda x: 0.9 * x,  # Discrete system x[k+1] = 0.9*x[k]
        dim,
        domain,
        discrete=True
    )
    
    states_discrete, _ = discrete_sampler.simulate_trajectory(x0, steps)
    
    # For discrete system x[k+1] = 0.9*x[k], check analytical solution
    # x[k] = 0.9^k * x0
    for i in range(1, steps + 1):
        expected_state = x0 * (0.9 ** i)
        assert np.allclose(states_discrete[i], expected_state, rtol=1e-6)

def test_seed_reproducibility():
    """Test reproducibility with random seed."""
    dim = 2
    domain = (np.array([-1, -1]), np.array([1, 1]))
    batch_size = 100
    seed = 42
    
    # Create two samplers with the same seed
    sampler1 = TrajectorySampler(linear_system, dim, domain, batch_size=batch_size, seed=seed)
    sampler2 = TrajectorySampler(linear_system, dim, domain, batch_size=batch_size, seed=seed)
    
    # Get batches
    x1, _ = sampler1.random_batch()
    x2, _ = sampler2.random_batch()
    
    # Should be identical
    assert np.array_equal(x1, x2)
    
    # Create sampler with different seed
    sampler3 = TrajectorySampler(linear_system, dim, domain, batch_size=batch_size, seed=seed+1)
    x3, _ = sampler3.random_batch()
    
    # Should be different
    assert not np.array_equal(x1, x3)
