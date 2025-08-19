"""
Strassen helper functions with Numba JIT optimization
"""
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    NUMBA_AVAILABLE = False


@njit(fastmath=True, cache=True, inline='always')
def fused_add4(X, Y, Z, W, out):
    """Compute out = X + Y - Z + W in a single pass"""
    n = X.shape[0]
    for i in range(n):
        for j in range(n):
            out[i, j] = X[i, j] + Y[i, j] - Z[i, j] + W[i, j]


@njit(fastmath=True, cache=True, inline='always')
def fused_add3_subtract(X, Y, Z, out):
    """Compute out = X - Y + Z in a single pass"""
    n = X.shape[0]
    for i in range(n):
        for j in range(n):
            out[i, j] = X[i, j] - Y[i, j] + Z[i, j]


@njit(fastmath=True, cache=True, inline='always')
def fused_add2(X, Y, out):
    """Compute out = X + Y"""
    n = X.shape[0]
    for i in range(n):
        for j in range(n):
            out[i, j] = X[i, j] + Y[i, j]


def strassen_combine_jit(P1, P2, P3, P4, P5, P6, P7):
    """
    Combine Strassen products using JIT-compiled functions
    Returns C11, C12, C21, C22
    """
    # Allocate outputs
    C11 = np.empty_like(P1)
    C12 = np.empty_like(P1)
    C21 = np.empty_like(P1)
    C22 = np.empty_like(P1)
    
    if NUMBA_AVAILABLE:
        # C11 = P1 + P4 - P5 + P7
        fused_add4(P1, P4, P5, P7, C11)
        
        # C12 = P3 + P5
        fused_add2(P3, P5, C12)
        
        # C21 = P2 + P4
        fused_add2(P2, P4, C21)
        
        # C22 = P1 - P2 + P3 + P6
        fused_add4(P1, P3, P2, P6, C22)
    else:
        # Fallback to NumPy operations
        np.add(P1, P4, out=C11)
        np.subtract(C11, P5, out=C11)
        np.add(C11, P7, out=C11)
        
        np.add(P3, P5, out=C12)
        
        np.add(P2, P4, out=C21)
        
        np.subtract(P1, P2, out=C22)
        np.add(C22, P3, out=C22)
        np.add(C22, P6, out=C22)
    
    return C11, C12, C21, C22


__all__ = ['fused_add4', 'fused_add3_subtract', 'fused_add2', 'strassen_combine_jit', 'NUMBA_AVAILABLE']
