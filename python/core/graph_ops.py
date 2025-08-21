# D:\Dev\kha\python\core\graph_ops.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Tuple, Optional

import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None  # we'll gracefully fall back to dense ops when needed


def _require_scipy():
    if sp is None:
        raise RuntimeError("scipy.sparse is required for sparse Laplacian ops. Install scipy.")


def normalize_laplacian_from_adjacency(A) -> "sp.csr_matrix | np.ndarray":
    """
    Symmetric normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    Works with scipy.sparse or dense np.ndarray.
    """
    if sp is not None and sp.issparse(A):
        d = np.asarray(A.sum(axis=1)).ravel()
        d_inv_sqrt = np.zeros_like(d)
        nz = d > 0
        d_inv_sqrt[nz] = 1.0 / np.sqrt(d[nz])
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        I = sp.eye(A.shape[0], format="csr")
        L = I - (D_inv_sqrt @ (A @ D_inv_sqrt))
        return L.tocsr()
    else:
        d = A.sum(axis=1)
        d_inv_sqrt = np.zeros_like(d)
        nz = d > 0
        d_inv_sqrt[nz] = 1.0 / np.sqrt(d[nz])
        D_inv_sqrt = np.diag(d_inv_sqrt)
        n = A.shape[0]
        I = np.eye(n)
        return I - D_inv_sqrt @ A @ D_inv_sqrt


def save_laplacian(L, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if sp is not None and sp.issparse(L):
        sp.save_npz(path, L.tocsr())
    else:
        np.save(path.with_suffix(".npy"), L)


def load_laplacian(path: str | Path):
    """
    Load Laplacian from .npz (scipy sparse) or .npy (dense).
    Prefers .npz, falls back to .npy.
    """
    path = Path(path)
    if path.suffix == ".npz" and path.exists():
        _require_scipy()
        return sp.load_npz(path)
    if path.suffix == ".npy" and path.exists():
        return np.load(path)
    # Try both extensions
    if path.with_suffix(".npz").exists():
        _require_scipy()
        return sp.load_npz(path.with_suffix(".npz"))
    if path.with_suffix(".npy").exists():
        return np.load(path.with_suffix(".npy"))
    raise FileNotFoundError(f"Laplacian not found at {path}(.npz|.npy)")


def adjacency_from_edgelist(n: int, edges: Iterable[Tuple[int, int, float]]):
    """
    Build symmetric weighted adjacency from (i, j, w) edges for i,j in [0,n).
    """
    _require_scipy()
    rows, cols, vals = [], [], []
    for i, j, w in edges:
        rows += [i, j]
        cols += [j, i]
        vals += [w, w]
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    A.sum_duplicates()
    return A


def _spmv(L, x):
    """Helper for sparse/dense matrix-vector multiply"""
    return (L @ x) if (sp is not None and sp.issparse(L)) else L.dot(x)


def spectral_radius(L, k: int = 10) -> float:
    """Approximate ||L||_2 by power iterations (cheap upper bound)."""
    n = L.shape[0]
    x = np.random.randn(n).astype(np.float64)
    x /= max(1e-12, np.linalg.norm(x))
    nrm = 0.0
    for _ in range(k):
        y = _spmv(L, x)
        nrm = float(np.linalg.norm(y))
        if nrm < 1e-12: break
        x = y / nrm
    return nrm


def symmetric_psd_sanity(L, atol_sym: float = 1e-6, trials: int = 4) -> None:
    """
    Cheap guards: symmetry and non-negativity by randomized Rayleigh quotients.
    Raises ValueError on violation.
    """
    # Symmetry check
    if sp is not None and sp.issparse(L):
        LT = L.T.tocsr()
        diff = (L - LT)
        sym_err = float(np.sum(np.abs(diff.data))) if diff.nnz else 0.0
        if sym_err > atol_sym:
            raise ValueError(f"Laplacian symmetry check failed (|L-L^T|_1={sym_err})")
    else:
        if not np.allclose(L, L.T, atol=atol_sym):
            raise ValueError("Laplacian symmetry check failed (dense)")
    
    # PSD check (randomized)
    n = L.shape[0]
    for _ in range(trials):
        v = np.random.randn(n)
        q = float(v @ _spmv(L, v))
        if q < -1e-8:
            raise ValueError(f"Laplacian not PSD by randomized check (v^T L v = {q})")
