from __future__ import annotations
from typing import Callable
import numpy as np

"""
External integrator hooks for FractalSolitonMemory.
This module is optional; if present, FSM will prefer `rk4_step` when 
`method="auto"` or `"rk4-ext"`.

Path: D:\\Dev\\kha\\python\\core\\fsm_lattice_integration.py
"""

Array = np.ndarray

def rk4_step(y: Array, dt: float, deriv: Callable[[Array], Array]) -> Array:
    """Classic explicit RK4. `deriv(y)` returns dy/dt with same shape as `y`.
    Deterministic, allocation-minimal.
    """
    k1 = deriv(y)
    k2 = deriv(y + 0.5 * dt * k1)
    k3 = deriv(y + 0.5 * dt * k2)
    k4 = deriv(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
