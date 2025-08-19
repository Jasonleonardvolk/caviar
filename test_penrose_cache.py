#!/usr/bin/env python3
"""
Test Penrose caching performance
"""
import os
import time
import numpy as np

os.environ["TORI_ENABLE_EXOTIC"] = "1"

from python.core.hot_swap_laplacian import HotSwappableLaplacian
from python.core.penrose_microkernel_v2 import multiply

print("Testing Penrose microkernel caching...")
print("=" * 60)

# Build Penrose Laplacian
hot_swap = HotSwappableLaplacian(
    initial_topology="penrose",
    lattice_size=(20, 20),
    enable_experimental=True,
)

L = hot_swap.graph_laplacian
print(f"Laplacian built: {L.shape}")

# Test multiple calls
sizes = [256, 512, 1024]
for i, n in enumerate(sizes):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    
    t0 = time.perf_counter()
    C, info = multiply(A, B, L)
    t1 = time.perf_counter()
    
    elapsed = (t1 - t0) * 1000  # ms
    
    print(f"\nCall {i+1} (n={n}):")
    print(f"  Time: {elapsed:.1f} ms")
    print(f"  Info: {info}")
    
    if i == 0 and elapsed > 100:
        print("  (First call includes eigensolve)")
    elif elapsed > 100:
        print("  WARNING: Still slow - caching might not be working!")

print("\n" + "=" * 60)
print("If caching works, only the first call should be slow.")
