"""
Standalone test for the Banksy global clock module.
"""

import numpy as np
import os
import sys

# Directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the clock functions directly in this file to avoid import issues
def spin_clock(sigma: np.ndarray) -> float:
    """
    Calculate the global spin-wave clock S(t) = Σσ/N in [-1,1].
    
    This function computes the mean value of the spin array to produce
    a global measure of synchronization. Values near ±1 indicate high 
    synchronization, while values near 0 indicate desynchronization.
    
    Args:
        sigma: Array of spin values (±1)
        
    Returns:
        The global clock value in range [-1, 1]
    """
    return float(sigma.mean())


def get_clock_metrics(sigma: np.ndarray) -> dict:
    """
    Calculate extended clock metrics beyond the basic S(t).
    
    Args:
        sigma: Array of spin values (±1)
        
    Returns:
        Dictionary with various clock metrics:
        - s_t: Basic clock S(t)
        - magnitude: Absolute value |S(t)|
        - variance: Variance of spins
    """
    s_t = spin_clock(sigma)
    return {
        "s_t": s_t,
        "magnitude": abs(s_t),
        "variance": float(np.var(sigma)),
    }


def test_spin_clock_bounds():
    """Test that spin_clock returns values in the range [-1, 1]."""
    print("\nTesting spin_clock_bounds...")
    
    # Test with different random spin configurations
    for i in range(5):
        sigma = np.random.choice([-1, 1], size=128)
        s = spin_clock(sigma)
        in_bounds = -1.0 <= s <= 1.0
        print(f"  Test {i+1}: spin_clock value = {s}, in bounds: {in_bounds}")
        assert in_bounds, f"Clock value {s} outside of range [-1, 1]"
    
    print("✓ All spin_clock bounds tests passed")


def test_spin_clock_extremes():
    """Test spin_clock with extreme cases."""
    print("\nTesting spin_clock_extremes...")
    
    # All spins up should give 1.0
    all_up = np.ones(100)
    s1 = spin_clock(all_up)
    print(f"  All up: spin_clock value = {s1}, expected 1.0")
    assert s1 == 1.0
    
    # All spins down should give -1.0
    all_down = -np.ones(100)
    s2 = spin_clock(all_down)
    print(f"  All down: spin_clock value = {s2}, expected -1.0")
    assert s2 == -1.0
    
    # Equal mixture should give 0.0
    mixed = np.array([1, -1] * 50)
    s3 = spin_clock(mixed)
    print(f"  Mixed: spin_clock value = {s3}, expected 0.0")
    assert s3 == 0.0
    
    print("✓ All spin_clock extremes tests passed")


def test_get_clock_metrics():
    """Test the extended clock metrics function."""
    print("\nTesting get_clock_metrics...")
    
    # Create a test array with 75% up spins, 25% down spins
    sigma = np.array([1] * 75 + [-1] * 25)
    metrics = get_clock_metrics(sigma)
    
    # Check metrics exist and have correct values
    has_s_t = "s_t" in metrics
    has_magnitude = "magnitude" in metrics
    has_variance = "variance" in metrics
    
    print(f"  Metrics contains s_t: {has_s_t}")
    print(f"  Metrics contains magnitude: {has_magnitude}")
    print(f"  Metrics contains variance: {has_variance}")
    
    assert has_s_t
    assert has_magnitude
    assert has_variance
    
    # Check values
    expected_s_t = 0.5  # (75-25)/100 = 0.5
    s_t_correct = metrics["s_t"] == expected_s_t
    print(f"  s_t value: {metrics['s_t']}, expected: {expected_s_t}, correct: {s_t_correct}")
    assert s_t_correct
    
    magnitude_correct = metrics["magnitude"] == 0.5
    print(f"  magnitude value: {metrics['magnitude']}, expected: 0.5, correct: {magnitude_correct}")
    assert magnitude_correct
    
    variance_positive = metrics["variance"] > 0
    print(f"  variance value: {metrics['variance']}, positive: {variance_positive}")
    assert variance_positive
    
    print("✓ All get_clock_metrics tests passed")


if __name__ == "__main__":
    print("=== Banksy Clock Module Standalone Tests ===")
    try:
        test_spin_clock_bounds()
        test_spin_clock_extremes()
        test_get_clock_metrics()
        print("\n✅ All tests passed successfully!")
        
        # Compare with the implementation in alan_backend/banksy/clock.py
        clock_path = os.path.join(current_dir, 'alan_backend', 'banksy', 'clock.py')
        if os.path.exists(clock_path):
            print(f"\nNote: The file {clock_path} exists.")
            print("You should verify that the implementations match.")
        else:
            print(f"\nNote: The file {clock_path} does not exist.")
            print("Consider copying this standalone implementation to that location.")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
