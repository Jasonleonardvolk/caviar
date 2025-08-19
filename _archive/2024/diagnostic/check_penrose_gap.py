#!/usr/bin/env python3
"""
Quick sanity check for Penrose spectral gap
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Force experimental
os.environ["TORI_ENABLE_EXOTIC"] = "1"

from python.core.exotic_topologies import build_penrose_laplacian
import scipy.sparse.linalg as sla
import numpy as np

print("Building Penrose Laplacian with Peierls flux...")
L = build_penrose_laplacian()
print(f"Laplacian shape: {L.shape}")
print(f"Laplacian dtype: {L.dtype}")

print("\nComputing eigenvalues with 'SM' (smallest magnitude)...")
# Get 6 smallest magnitude eigenvalues
ev = sla.eigsh(L, k=min(6, L.shape[0]-1), which="SM", return_eigenvectors=False)

print("\nEigenvalues:")
for i, lam in enumerate(sorted(ev)):
    print(f"λ{i} = {lam:.6f}")

# Find first non-zero eigenvalue
nz = [lam for lam in ev if lam > 1e-8]
if nz:
    gap = min(nz)
    print(f"\nSpectral gap (first non-zero eigenvalue) = {gap:.6f}")
    if gap > 0.2:
        print("✓ Gap is healthy (> 0.2)")
    elif gap > 0.1:
        print("✓ Gap is good (> 0.1)")
    else:
        print("✗ Gap is small but should work")
else:
    print("\n✗ No non-zero eigenvalues found!")
