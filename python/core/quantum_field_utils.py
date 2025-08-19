from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# File: {PROJECT_ROOT}\python\core\quantum_field_utils.py

import numpy as np
from typing import Tuple

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

@jit(nopython=True, parallel=True, cache=True)
def compute_phase_and_curvature_gradients(
    phase_field: np.ndarray,
    curvature_field: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute phase and curvature gradients using central differences.

    Parameters:
        phase_field (np.ndarray): 2D array of ψ-phase values (radians, normalized [−π, π])
        curvature_field (np.ndarray): 2D array of geometric curvature (Kretschmann scalar)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            phase_gradient (H x W x 2): ∇ψ for coupling
            curvature_gradient (H x W x 2): ∇𝒦 for force feedback

    Units:
        ∇ψ: radians per lattice unit
        ∇𝒦: scalar field differential
    """
    rows, cols = phase_field.shape
    phase_grad = np.zeros((rows, cols, 2))
    curvature_grad = np.zeros((rows, cols, 2))

    for i in prange(1, rows - 1):
        for j in prange(1, cols - 1):
            # Central differences (i ± 1, j ± 1)
            phase_grad[i, j, 0] = (phase_field[i+1, j] - phase_field[i-1, j]) * 0.5
            phase_grad[i, j, 1] = (phase_field[i, j+1] - phase_field[i, j-1]) * 0.5

            curvature_grad[i, j, 0] = (curvature_field[i+1, j] - curvature_field[i-1, j]) * 0.5
            curvature_grad[i, j, 1] = (curvature_field[i, j+1] - curvature_field[i, j-1]) * 0.5

    return phase_grad, curvature_grad
