#!/usr/bin/env python3
"""
Verify rank-14 numerical stability
"""
import os
import numpy as np
from numpy.linalg import norm

os.environ["TORI_ENABLE_EXOTIC"] = "1"

from python.core.hot_swap_laplacian import HotSwappableLaplacian
from python.core.penrose_microkernel_v2 import multiply, RANK

print(f"PENROSE RANK-{RANK} STABILITY VERIFICATION")
print("=" * 60)

# Build Penrose Laplacian
hot_swap = HotSwappableLaplacian(
    initial_topology="penrose",
    lattice_size=(20, 20),
    enable_experimental=True,
)

L = hot_swap.graph_laplacian
n = 512  # Test size

# Random matrices
A = np.random.rand(n, n)
B = np.random.rand(n, n)

# Exact and Penrose results
C_exact = A @ B
C_penrose, info = multiply(A, B, L)

# Check relative error
error = norm(C_penrose - C_exact) / norm(C_exact)
gap = info.get('spectral_gap', 0)

print(f"\nResults for n={n}:")
print(f"  Spectral gap: {gap:.4f}")
print(f"  Relative error: {error:.2e}")
print(f"  Expected leakage: ~{gap*RANK:.1e}")

# Check Laplacian action (optional)
# This verifies the projector quality
L_slice = L[:n, :n].toarray()  # Dense for this test
LC_norm = norm(L_slice @ C_penrose) / norm(C_penrose)
print(f"  ||L·C||/||C||: {LC_norm:.3f} (should be < {1.5*gap:.3f})")

print("\n" + "=" * 60)
if error < 0.01 and LC_norm < 1.5*gap:
    print("✓ RANK-14 IS NUMERICALLY STABLE")
    print("  Safe to use for production benchmarks")
else:
    print("⚠ Consider staying with rank-16 for stability")
