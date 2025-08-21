# D:\Dev\kha\python\core\gpu_ops.py
from __future__ import annotations
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False
    cp = None; cpx = None

def to_gpu_L(L):
    if not GPU_AVAILABLE: raise RuntimeError("CuPy not available")
    if hasattr(L, "tocsr"):
        return cpx.csr_matrix(L)
    return cpx.csr_matrix(L)

def resonance_scores_gpu(L_gpu, n: int, q_idx: int, T: int = 20, dt: float = 0.2, kappa: float = 1.0, lam: float = 0.0):
    psi = cp.zeros(n, dtype=cp.complex128); psi[q_idx] = 1.0 + 0.0j
    for _ in range(T):
        grad = kappa * (L_gpu @ psi) + lam * (cp.abs(psi) ** 2) * psi
        psi = psi - dt * grad
    return cp.asnumpy(cp.abs(psi))
