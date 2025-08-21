# D:\Dev\kha\python\core\fractional_laplacian.py
from __future__ import annotations
import numpy as np
try:
    import scipy.sparse as sp
except Exception:
    sp = None

from python.core.graph_ops import spectral_radius

def _spmv(L, x):
    if sp is not None and sp.issparse(L): return L @ x
    return L.dot(x)

def chebyshev_apply_power(L, x, alpha: float = 0.5, m: int = 30):
    """
    Approximate y ≈ L^alpha x via Chebyshev on spectrum [0, rho].
    We map λ ∈ [0, rho] to s ∈ [-1,1], approximate f(λ)=λ^α.
    """
    rho = spectral_radius(L, k=10)
    if rho <= 0: return x.copy()
    
    # scale: Ls = (2/rho) L - I  with eigenvalues in [-1,1]
    def Ls_mv(v):
        return (2.0 / rho) * _spmv(L, v) - v

    # Chebyshev recurrence
    def T_k_apply(k, v0):
        if k == 0: return v0
        if k == 1: return Ls_mv(v0)
        Tkm2 = v0; Tkm1 = Ls_mv(v0)
        for _ in range(2, k+1):
            Tk = 2.0 * Ls_mv(Tkm1) - Tkm2
            Tkm2, Tkm1 = Tkm1, Tk
        return Tkm1

    # coefficients c_k from sampling at θ_j points
    j = np.arange(0, m+1)
    theta = (np.pi * (j + 0.5)) / (m + 1)
    s = np.cos(theta)
    lam = (rho * (s + 1.0)) / 2.0
    f = lam ** alpha
    c = (2.0 / (m + 1)) * np.cos(np.outer(j, theta)) @ f
    c[0] *= 0.5  # a_0/2

    y = c[0] * x
    for k in range(1, m+1):
        y += c[k] * T_k_apply(k, x)
    return y
