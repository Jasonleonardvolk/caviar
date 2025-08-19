#!/usr/bin/env python3
"""
Exotic Topology Builders V2 - Supporting Large Penrose Tilings
─────────────────────────────────────────────────────────────
Enhanced Penrose Laplacian builder that can generate 5200+ nodes
"""
from __future__ import annotations
import math
from typing import Tuple

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

PHI = (1 + 5 ** 0.5) / 2  # golden ratio


def build_penrose_laplacian_large(
    target_nodes: int = 5200,
    connect_radius: float = 0.75,
) -> sp.csr_matrix:
    """
    Build a Penrose Laplacian with at least target_nodes vertices.
    Automatically scales the patch size to achieve the target.
    """
    # Estimate patch size needed
    # Empirically, patch_size (m,m) gives roughly 0.8 * m^2 nodes
    estimated_m = int(np.sqrt(target_nodes / 0.8)) + 2
    
    print(f"Building Penrose tiling with target {target_nodes} nodes...")
    print(f"Estimated patch size: ({estimated_m}, {estimated_m})")
    
    # Five star directions for de Bruijn grid
    angles = [2 * math.pi * k / 5 for k in range(5)]
    u = np.array([[math.cos(a), math.sin(a)] for a in angles])

    # Keep increasing patch size until we have enough nodes
    for m in range(estimated_m, estimated_m + 10):
        verts: dict[tuple, np.ndarray] = {}
        
        for k in range(-m, m + 1):
            for l in range(-m, m + 1):
                for i in range(5):
                    j = (i + 2) % 5          # non-parallel line pair
                    denom = np.cross(u[i], u[j])
                    if abs(denom) < 1e-9:
                        continue
                    pt = (k * u[i] + l * u[j]) / denom
                    verts.setdefault(tuple(np.round(pt, 5)), pt)
        
        if len(verts) >= target_nodes:
            print(f"Generated {len(verts)} nodes with patch size ({m}, {m})")
            break
    else:
        print(f"Warning: Only generated {len(verts)} nodes, less than target {target_nodes}")

    coords = np.vstack(list(verts.values()))
    tree = cKDTree(coords)
    
    # Build connectivity with adaptive radius
    radius = connect_radius
    attempts = 0
    while attempts < 20:  # Safety limit
        pairs = np.array(list(tree.query_pairs(r=radius)), dtype=np.int32)

        if len(pairs) == 0:
            radius *= 1.5
            attempts += 1
            continue

        # Build adjacency matrix
        data = np.ones(len(pairs) * 2, dtype=np.float64)
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        A = sp.coo_matrix((data, (rows, cols)),
                          shape=(coords.shape[0],) * 2).tocsr()

        n_comp, _ = sp.csgraph.connected_components(A, directed=False)
        if n_comp == 1:
            print(f"Info: radius grown to {radius:.2f} — graph connected")
            print(f"Graph has {A.shape[0]} nodes, {A.nnz//2} edges")
            break
        
        radius *= 1.3
        attempts += 1

    # Build complex Laplacian with Peierls flux
    L = sp.csgraph.laplacian(A, normed=False).tocoo().astype(np.complex128)
    
    # χ = 1 flux : φ = 2π/N
    phi = 2 * np.pi / L.shape[0]
    L.data *= np.exp(1j * phi * (L.col - L.row))
    
    # Hermitise, then CSR
    L = ((L + L.T.conj()) * 0.5).tocsr()
    
    return L
