# D:\Dev\kha\python\bench\baselines.py
from __future__ import annotations
import numpy as np
try:
    import scipy.sparse as sp
    from scipy.sparse.linalg import expm_multiply
except Exception:
    sp = None
    expm_multiply = None

def _to_csr(A):
    if sp is not None and sp.issparse(A): return A.tocsr()
    return sp.csr_matrix(A) if sp is not None else None

def row_normalize(A):
    if sp is not None and sp.issparse(A):
        d = np.asarray(A.sum(axis=1)).ravel()
        d[d == 0.0] = 1.0
        Dinv = sp.diags(1.0 / d)
        return Dinv @ A
    else:
        d = A.sum(axis=1, keepdims=True)
        d[d == 0.0] = 1.0
        return A / d

def adjacency_from_L(L):
    # For normalized Laplacian L = I - D^{-1/2} A D^{-1/2}, we use A_tilde = I - L (proxy),
    # then row-normalize to obtain a walk matrix. This is an approximation if only L is available.
    if sp is not None and sp.issparse(L):
        I = sp.eye(L.shape[0], format="csr")
        Atil = (I - L).tocsr()
        Atil.data[Atil.data < 0.0] = 0.0
        return row_normalize(Atil)
    else:
        I = np.eye(L.shape[0])
        Atil = I - L
        Atil[Atil < 0.0] = 0.0
        row_sums = Atil.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return Atil / row_sums

def ppr_scores(L, q_idx: int, alpha: float = 0.85, tol: float = 1e-8, maxit: int = 100) -> np.ndarray:
    P = adjacency_from_L(L)  # row-stochastic
    n = P.shape[0]
    e_q = np.zeros(n); e_q[q_idx] = 1.0
    r = e_q.copy()
    x = np.zeros(n)
    for _ in range(maxit):
        x_new = (1 - alpha) * e_q + alpha * (P.T @ x)
        if np.linalg.norm(x_new - x, 1) < tol:
            x = x_new; break
        x = x_new
    return x

def heat_kernel_scores(L, q_idx: int, t: float = 0.5) -> np.ndarray:
    n = L.shape[0]
    e_q = np.zeros(n); e_q[q_idx] = 1.0
    if expm_multiply is not None and sp is not None and sp.issparse(L):
        return np.abs(expm_multiply((-t) * L, e_q))
    # Fallback: truncated series exp(-tL) e_q ≈ sum_{k=0..m} (-t)^k/k! L^k e_q
    m = 10
    out = e_q.copy()
    v = e_q.copy()
    coef = 1.0
    for k in range(1, m+1):
        v = (L @ v) if (sp is not None and sp.issparse(L)) else L.dot(v)
        coef *= (-t) / k
        out += coef * v
    return np.abs(out)

def simrank_lite_scores(L, q_idx: int, C: float = 0.6, iters: int = 10) -> np.ndarray:
    # Lightweight SimRank using neighbors from approximated adjacency
    P = adjacency_from_L(L)  # row-stochastic (use as neighbor proxy)
    if sp is not None and sp.issparse(P):
        P = P.tocsr()
        P = P / (P.sum(axis=1) + 1e-12)
    n = P.shape[0]
    S = np.eye(n)
    for _ in range(iters):
        # S_new = C * P^T S P ; diag(S)=1
        if sp is not None:
            S = C * (P.T @ (S @ P))
        else:
            S = C * (P.T @ S @ P)
        np.fill_diagonal(S, 1.0)
    return S[:, q_idx]
