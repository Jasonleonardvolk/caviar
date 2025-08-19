#!/usr/bin/env python3
"""Test coupling matrix and Hebbian learning."""

from python.core.coupling_matrix import CouplingMatrix
import numpy as np


def test_strengthen_and_decay():
    """Test Hebbian strengthening and decay."""
    C = CouplingMatrix(3)
    
    # Initial state should be all zeros (or k_init value)
    assert np.allclose(C.K, 0.0)
    
    # Strengthen connection between 0 and 1
    C.strengthen(0, 1, 0.5)
    assert np.isclose(C.K[0, 1], 0.5)
    assert np.isclose(C.K[1, 0], 0.5)  # Should be symmetric
    
    # Apply decay
    C.decay(rate=0.1)
    assert C.K[0, 1] < 0.5
    assert np.isclose(C.K[0, 1], 0.45)  # 0.5 * (1 - 0.1)


def test_resize():
    """Test resizing coupling matrix."""
    C = CouplingMatrix(2, k_init=0.1)
    
    # Set some specific values
    C.K[0, 1] = 0.5
    C.K[1, 0] = 0.5
    
    # Resize to larger
    C.resize(4, k_init=0.2)
    
    # Old values should be preserved
    assert np.isclose(C.K[0, 1], 0.5)
    assert np.isclose(C.K[1, 0], 0.5)
    
    # New entries should have k_init value
    assert np.isclose(C.K[2, 3], 0.2)
    assert np.isclose(C.K[3, 2], 0.2)
    
    # Shape should be correct
    assert C.K.shape == (4, 4)


def test_repeated_strengthening():
    """Test that repeated strengthening accumulates."""
    C = CouplingMatrix(3)
    
    # Strengthen same connection multiple times
    for _ in range(10):
        C.strengthen(0, 2, 0.1)
    
    assert np.isclose(C.K[0, 2], 1.0)
    assert np.isclose(C.K[2, 0], 1.0)  # Symmetric


def test_global_decay():
    """Test that decay affects all connections."""
    C = CouplingMatrix(3, k_init=1.0)
    
    # All connections start at 1.0
    assert np.allclose(C.K, 1.0)
    
    # Apply decay
    C.decay(rate=0.5)
    
    # All should be reduced by half
    assert np.allclose(C.K, 0.5)


if __name__ == "__main__":
    test_strengthen_and_decay()
    test_resize()
    test_repeated_strengthening()
    test_global_decay()
    print("All coupling matrix tests passed!")
