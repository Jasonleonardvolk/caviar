from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
import numpy as np

try:
    import scipy.sparse as sp  # type: ignore
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

Array = np.ndarray


def _as_dense(M):
    if HAS_SCIPY and sp.issparse(M):
        return M.toarray()
    return np.asarray(M)


def load_laplacian_from_npz(path: str):
    data = np.load(path, allow_pickle=True)
    if all(k in data.files for k in ("data","indices","indptr","shape")):
        if not HAS_SCIPY:
            raise RuntimeError("scipy required to load sparse Laplacian")
        return sp.csr_matrix((data["data"], data["indices"], data["indptr"]), shape=tuple(data["shape"]))
    if "L" in data.files:
        return np.asarray(data["L"])
    raise ValueError("Unsupported Laplacian npz schema")


def infer_adjacency_from_laplacian(
    L, *,
    mode: Literal["auto","combinatorial","normalized"] = "auto",
    clip_neg: bool = True,
) -> Array:
    """Infer adjacency A from Laplacian L.
    - combinatorial: L = D - A  =>  A = D - L
    - normalized  : L = I - D^{-1/2} A D^{-1/2}  =>  Â = I - L (symmetric); degrees ≈ Â 1; return A = Â (scaled)
    For mode="auto": if mean(diag(L)) ~ 1 -> assume normalized; else combinatorial.
    """
    M = _as_dense(L)
    n = M.shape[0]
    d = np.diag(M)
    if mode == "auto":
        mode = "normalized" if np.allclose(np.mean(d), 1.0, atol=1e-3) else "combinatorial"
    if mode == "combinatorial":
        D = np.diag(d)
        A = D - M
    else:
        A = np.eye(n) - M
    if clip_neg:
        A = np.where(A < 0, 0.0, A)
    # zero diag, symmetrize
    np.fill_diagonal(A, 0.0)
    A = 0.5*(A + A.T)
    return A


def compute_degrees(A: Array) -> Array:
    return np.sum(A, axis=1)


def compute_forman_ricci_node(A: Array) -> Array:
    """Forman–Ricci curvature per node (unweighted/simple approximation).
    For an undirected simple graph, edge curvature: F(u,v) = 4 - deg(u) - deg(v).
    Node curvature: K(u) = sum_{v in N(u)} F(u,v).
    For weighted A, we weight edges by A[u,v]. This is a pragmatic estimator.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    deg = compute_degrees(A)
    # Edge curvature matrix F_e ~ 4 - d_u - d_v
    Du = deg.reshape(-1,1)
    Dv = deg.reshape(1,-1)
    F_edge = (4.0 - (Du + Dv))
    # weight by adjacency
    F_edge_w = F_edge * (A > 0).astype(float)
    # Node curvature = sum over incident edges (weighted by A > 0)
    K_node = np.sum(F_edge_w, axis=1)
    return K_node


def compute_kretschmann_like(K_node: Array, A: Array) -> Array:
    """Graph analogue: per-node sum of squared incident edge curvatures.
    We approximate as neighborhood L2 of node curvature: Kretz(u) = (K(u))^2 + sum_{v in N(u)} (K(v))^2 * w(u,v)_norm.
    """
    A = np.asarray(A, dtype=float)
    deg = compute_degrees(A) + 1e-12
    w = A / deg[:,None]  # row-normalized weights
    K2 = (K_node**2)
    neigh = w @ K2
    return K2 + neigh


def compute_mean_curvature_like(A: Array, signal: Optional[Array]=None) -> Array:
    """Proxy for mean curvature using a graph Laplacian acting on a smooth signal.
    If `signal` is None, use the node curvature field K_node as the signal and apply the (combinatorial) Laplacian.
    """
    n = A.shape[0]
    deg = compute_degrees(A)
    Lc = np.diag(deg) - A
    if signal is None:
        signal = compute_forman_ricci_node(A)
    return Lc @ signal


@dataclass
class CurvatureOutputs:
    ricci: Array
    kretschmann: Array
    mean_curvature: Array


def compute_curvature_fields_from_L(
    L, *, mode: Literal["auto","combinatorial","normalized"] = "auto"
) -> CurvatureOutputs:
    """High-level convenience: infer A from L, compute discrete curvature proxies."""
    A = infer_adjacency_from_laplacian(L, mode=mode)
    ricci = compute_forman_ricci_node(A)
    kretsch = compute_kretschmann_like(ricci, A)
    mean_curv = compute_mean_curvature_like(A, signal=ricci)
    # standardize to zero-mean, unit-variance to be safe for encoders
    def _std(z):
        z = z.astype(float)
        return (z - np.mean(z)) / (np.std(z) + 1e-12)
    return CurvatureOutputs(ricci=_std(ricci), kretschmann=_std(kretsch), mean_curvature=_std(mean_curv))


def save_curvature_fields(out_dir: str, fields: CurvatureOutputs) -> Tuple[str,str,str]:
    import os
    os.makedirs(out_dir, exist_ok=True)
    r = os.path.join(out_dir, "ricci.npy")
    k = os.path.join(out_dir, "kretschmann.npy")
    m = os.path.join(out_dir, "mean_curvature.npy")
    np.save(r, fields.ricci)
    np.save(k, fields.kretschmann)
    np.save(m, fields.mean_curvature)
    return r,k,m
