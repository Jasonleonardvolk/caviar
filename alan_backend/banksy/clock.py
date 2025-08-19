"""
Banksy Global Clock Module

Implements the global spin-wave clock S(t) that provides a measure
of global synchronization across the oscillator network.
"""

import numpy as np
from typing import Union, List, Tuple


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
