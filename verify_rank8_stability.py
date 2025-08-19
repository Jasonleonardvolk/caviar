#!/usr/bin/env python3
"""
Verify numerical stability of rank-8 Penrose projector
"""
import os
import numpy as np
from numpy.linalg import norm

os.environ["TORI_ENABLE_EXOTIC"] = "1"

from python.core.hot_swap_laplacian import HotSwappableLaplacian
from python.core.penrose_microkernel_v2 import multiply, RANK

print("PENROSE RANK-8 STABILITY CHECK")
print("=" * 60)
print(f"Current rank: {RANK}")

# Build Penrose Laplacian
hot_swap = HotSwappableLaplacian(
    initial_topology="penrose",
    lattice_size=(20, 20),
    enable_experimental=True,
)

L = hot_swap.graph_laplacian
print(f"Laplacian shape: {L.shape}")

# Test multiplication accuracy
sizes = [256, 512, 1024]
for n in sizes:
    print(f"\nTesting n={n}:")
    
    # Random test matrices
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    
    # Exact result
    C_exact = A @ B
    
    # Penrose result
    C_penrose, info = multiply(A, B, L)
    
    # Check accuracy
    error = norm(C_penrose - C_exact) / norm(C_exact)
    print(f"  Relative error: {error:.2e}")
    print(f"  Spectral gap: {info.get('spectral_gap', 0):.4f}")
    
    # Check spectral leakage (optional)
    # This would require converting C back through the Laplacian
    # For now, we just check the multiplication error
    
    if error > 0.1:  # 10% threshold
        print("  WARNING: High error - rank may be too low!")
    else:
        print("  ✓ Accuracy acceptable")

print("\n" + "=" * 60)
print("Recommendation:")
print("  If all errors < 10%, rank-8 is safe for world record attempt")
print("  If errors > 10%, consider rank-12 for stability")
