"""
Unit tests for the Banksy global clock module.
"""

import numpy as np
import pytest
from alan_backend.banksy.clock import spin_clock, get_clock_metrics


def test_spin_clock_bounds():
    """Test that spin_clock returns values in the range [-1, 1]."""
    # Test with different random spin configurations
    for _ in range(5):
        sigma = np.random.choice([-1, 1], size=128)
        s = spin_clock(sigma)
        assert -1.0 <= s <= 1.0, f"Clock value {s} outside of range [-1, 1]"


def test_spin_clock_extremes():
    """Test spin_clock with extreme cases."""
    # All spins up should give 1.0
    all_up = np.ones(100)
    assert spin_clock(all_up) == 1.0
    
    # All spins down should give -1.0
    all_down = -np.ones(100)
    assert spin_clock(all_down) == -1.0
    
    # Equal mixture should give 0.0
    mixed = np.array([1, -1] * 50)
    assert spin_clock(mixed) == 0.0


def test_get_clock_metrics():
    """Test the extended clock metrics function."""
    # Create a test array with 75% up spins, 25% down spins
    sigma = np.array([1] * 75 + [-1] * 25)
    metrics = get_clock_metrics(sigma)
    
    # Check metrics exist and have correct values
    assert "s_t" in metrics
    assert "magnitude" in metrics
    assert "variance" in metrics
    
    # Check values
    assert metrics["s_t"] == 0.5  # (75-25)/100 = 0.5
    assert metrics["magnitude"] == 0.5
    assert metrics["variance"] > 0  # Should be positive for non-uniform array


if __name__ == "__main__":
    pytest.main()
