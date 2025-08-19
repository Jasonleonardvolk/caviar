#!/usr/bin/env python3
"""
Diagnose Penrose connectivity issues
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from python.core.exotic_topologies import build_penrose_laplacian
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import numpy as np

print("Testing Penrose connectivity with different radii...\n")

for radius in [0.75, 0.85, 1.0, 1.2, 1.5, 2.0]:
    print(f"Connect radius = {radius}")
    
    # Build adjacency matrix from Laplacian
    L = build_penrose_laplacian(connect_radius=radius)
    
    # Extract adjacency from Laplacian: A = diag(L) - L
    D = sp.diags(L.diagonal())
    A = D - L
    
    # Check connectivity
    n_components, labels = sp.csgraph.connected_components(A, directed=False)
    print(f"  Components: {n_components}")
    
    if n_components == 1:
        # Get eigenvalues
        try:
            eigenvalues = sla.eigsh(L, k=min(10, L.shape[0]-1), which="SA", 
                                   return_eigenvectors=False, tol=1e-6)
            eigenvalues = np.sort(eigenvalues)
            print(f"  First 5 eigenvalues: {eigenvalues[:5]}")
            
            # Find spectral gap
            non_zero = [ev for ev in eigenvalues if ev > 1e-8]
            if non_zero:
                gap = non_zero[0]
                print(f"  Spectral gap: {gap:.6f}")
            else:
                print(f"  No significant eigenvalues found!")
        except Exception as e:
            print(f"  Error computing eigenvalues: {e}")
        
        print(f"  ✓ Connected graph achieved!")
        break
    else:
        print(f"  ✗ Still disconnected")
        
print("\nRecommendation: Use the smallest radius that gives a connected graph.")
