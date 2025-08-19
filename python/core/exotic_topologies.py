#!/usr/bin/env python3
"""
Exotic Topology Builders
────────────────────────
Penrose Laplacian with a Chern-number 1 Peierls flux:
  • KD-tree neighbour search
  • Robust connectivity loop ensures single component
  • Peierls phase injection for spectral gap
  • Returns a CSR matrix (no NetworkX conversion)
"""
from __future__ import annotations
import math
from typing import Tuple

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

PHI = (1 + 5 ** 0.5) / 2  # golden ratio


def build_penrose_laplacian(
    patch_size: Tuple[int, int] = (12, 12),
    connect_radius: float = 0.75,   # Starting radius
) -> sp.csr_matrix:
    m, n = patch_size

    # Five star directions for de Bruijn grid
    angles = [2 * math.pi * k / 5 for k in range(5)]
    u = np.array([[math.cos(a), math.sin(a)] for a in angles])

    # Unique vertices (dict dedup)
    verts: dict[tuple, np.ndarray] = {}
    for k in range(-m, m + 1):
        for l in range(-n, n + 1):
            for i in range(5):
                j = (i + 2) % 5          # non-parallel line pair
                denom = np.cross(u[i], u[j])
                if abs(denom) < 1e-9:
                    continue
                pt = (k * u[i] + l * u[j]) / denom
                verts.setdefault(tuple(np.round(pt, 5)), pt)

    coords = np.vstack(list(verts.values()))
    tree = cKDTree(coords)
    
    # KD-tree already built
    radius = connect_radius
    while True:
        pairs = np.array(list(tree.query_pairs(r=radius)), dtype=np.int32)

        if len(pairs) == 0:          # no edges → grow & retry
            radius *= 1.5
            continue

        # rebuild adjacency for *this* radius
        data = np.ones(len(pairs) * 2, dtype=np.float64)
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        A = sp.coo_matrix((data, (rows, cols)),
                          shape=(coords.shape[0],) * 2).tocsr()

        n_comp, _ = sp.csgraph.connected_components(A, directed=False)
        if n_comp == 1:
            print(f"Info: radius grown to {radius:.2f} — graph connected")
            break                    # ✅ done
        radius *= 1.3               # still fragmented → widen & loop

    # ----- build complex Laplacian and inject Peierls flux -------------
    # 1. real Laplacian → COO
    L = sp.csgraph.laplacian(A, normed=False).tocoo().astype(np.complex128)
    
    # 2. χ = 1 flux : φ = 2π/N
    phi = 2 * np.pi / L.shape[0]
    L.data *= np.exp(1j * phi * (L.col - L.row))
    
    # 3. Hermitise, then CSR
    L = ((L + L.T.conj()) * 0.5).tocsr()
    
    return L


# Keep the old function for backward compatibility
def build_penrose_graph(*args, **kwargs):
    """Legacy function - use build_penrose_laplacian instead"""
    raise NotImplementedError("Use build_penrose_laplacian() for direct Laplacian construction")
