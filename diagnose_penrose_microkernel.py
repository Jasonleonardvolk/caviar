#!/usr/bin/env python3
"""
Diagnose Penrose microkernel issues
"""
import os
import numpy as np
import scipy.sparse.linalg as sla

# Force experimental mode
os.environ["TORI_ENABLE_EXOTIC"] = "1"

from python.core.hot_swap_laplacian import HotSwappableLaplacian

print("PENROSE MICROKERNEL DIAGNOSTICS")
print("=" * 60)

# Build Penrose Laplacian
hot_swap = HotSwappableLaplacian(
    initial_topology="kagome",
    lattice_size=(20, 20),
    enable_experimental=True,
)

print("\n1. Building Penrose Laplacian...")
hot_swap.current_topology = "penrose"
hot_swap.graph_laplacian = hot_swap._build_laplacian("penrose")
print("   ✓ Built successfully")

L = hot_swap.graph_laplacian
print(f"\n2. Laplacian properties:")
print(f"   Shape: {L.shape}")
print(f"   Dtype: {L.dtype}")
print(f"   Non-zeros: {L.nnz}")

print("\n3. Testing eigenvalue computation...")
# Try the same approach as the microkernel
k = 2
max_k = 20
found_gap = False

while k <= max_k:
    print(f"\n   Trying k={k}...")
    try:
        ev = sla.eigsh(L, k=k, which="SA", return_eigenvectors=False, tol=1e-6)
        print(f"   Got {len(ev)} eigenvalues: {[f'{x:.6f}' for x in sorted(ev)]}")
        
        nz = [lam for lam in ev if lam > 1e-8]
        if nz:
            spectral_gap = float(min(nz))
            print(f"   ✓ Found spectral gap: {spectral_gap:.6f}")
            found_gap = True
            break
        else:
            print(f"   No non-zero eigenvalues yet")
    except Exception as e:
        print(f"   Error: {e}")
        break
    
    k += 2
    if k > L.shape[0] - 1:
        print(f"   Reached matrix size limit")
        break

if not found_gap:
    print("\n✗ No spectral gap found!")
else:
    print(f"\n✓ Spectral gap = {spectral_gap:.6f}")
    if spectral_gap > 1e-5:
        print("✓ Gap should pass microkernel threshold")
    else:
        print("✗ Gap too small for microkernel")

print("\n4. Testing small matrix multiplication...")
A = np.random.rand(64, 64)
B = np.random.rand(64, 64)

from python.core.penrose_microkernel_v2 import multiply
try:
    C, info = multiply(A, B, L)
    print(f"   ✓ Small matrix (64x64) worked: {info}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n5. Testing larger matrix...")
A = np.random.rand(256, 256)
B = np.random.rand(256, 256)
try:
    print("   Starting multiplication...")
    C, info = multiply(A, B, L)
    print(f"   ✓ Large matrix (256x256) worked: {info}")
except KeyboardInterrupt:
    print("   ✗ Interrupted - likely stuck in eigenvalue loop")
except Exception as e:
    print(f"   ✗ Error: {e}")
