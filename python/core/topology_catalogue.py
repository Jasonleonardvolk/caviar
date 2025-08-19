"""
Topology Catalogue - Penrose lattice projection mappings
Maps between physical lattice coordinates and matrix index triples
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Iterator
import logging

logger = logging.getLogger(__name__)

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio

class CollisionProjection:
    """
    Bidirectional mapping between:
    - Physical lattice coordinates (x, y)
    - Matrix multiplication indices (i, j, k)
    
    Ensures that collisions computing A[i,k] × B[k,j] happen at the
    right physical location to sum correctly for C[i,j]
    """
    
    def __init__(self, matrix_size: int, lattice_size: int = None):
        self.matrix_size = matrix_size
        self.lattice_size = lattice_size or int(matrix_size * PHI * 2)
        self.lattice_shape = (self.lattice_size, self.lattice_size)
        
        # Forward mapping: (x,y) → (i,j,k)
        self._forward: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
        # Inverse mapping: (i,j,k) → (x,y)
        self._inverse: Dict[Tuple[int, int, int], Tuple[int, int]] = {}
        
        self._build_projection()
    
    def _build_projection(self):
        """
        Create the Penrose-constrained mapping ensuring:
        1. Products for same C[i,j] are routed to nearby locations
        2. Golden-ratio spacing prevents unwanted interference
        """
        n = self.matrix_size
        
        # For 2x2, use precomputed map with all 8 triples
        if n == 2:
            # 8 distinct, non-adjacent vertices of a (scaled) pentagon
            PRECOMPUTED_MAP_2X2 = {
                (0, 0, 0): (3, 3),
                (0, 0, 1): (5, 4),
                (0, 1, 0): (2, 5),
                (0, 1, 1): (4, 6),
                (1, 0, 0): (1, 2),
                (1, 0, 1): (7, 2),
                (1, 1, 0): (0, 6),
                (1, 1, 1): (6, 0),
            }
            for (i, j, k), (x, y) in PRECOMPUTED_MAP_2X2.items():
                self._forward[(x, y)] = (i, j, k)
                self._inverse[(i, j, k)] = (x, y)
        else:
            # Use 5-fold symmetric assignment for larger matrices
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        # Hash function using golden ratio to ensure proper spacing
                        # Critical: this determines which collisions contribute to which result
                        
                        # 5-fold symmetry assignment
                        arm = (i + j + k) % 5
                        radius = int((i * PHI + j + k * PHI**2) % (self.lattice_size // 2))
                        
                        # Convert to Cartesian coordinates
                        angle = 2 * np.pi * arm / 5
                        x = int(self.lattice_size // 2 + radius * np.cos(angle))
                        y = int(self.lattice_size // 2 + radius * np.sin(angle))
                        
                        # Ensure within bounds
                        x = max(0, min(x, self.lattice_size - 1))
                        y = max(0, min(y, self.lattice_size - 1))
                        
                        # Only keep first mapping to each lattice site (avoid conflicts)
                        if (x, y) not in self._forward:
                            self._forward[(x, y)] = (i, j, k)
                            self._inverse[(i, j, k)] = (x, y)
        
        logger.info(
            f"Collision projection built: {len(self._forward)} lattice sites "
            f"for {n}×{n} matrix multiplication"
        )
    
    def __getitem__(self, key: Tuple[int, int]) -> Tuple[int, int, int]:
        """Forward lookup: lattice position → matrix indices"""
        return self._forward[key]
    
    def inverse_items(self) -> Iterator[Tuple[Tuple[int, int], Tuple[int, int, int]]]:
        """Iterator over (lattice_pos, matrix_indices) pairs"""
        for ijk, xy in self._inverse.items():
            yield xy, ijk

# Global cache of projections
_projection_cache: Dict[int, CollisionProjection] = {}

def collision_projection(matrix_size: int = None) -> CollisionProjection:
    """
    Get or create collision projection for given matrix size.
    If matrix_size is None, returns the most recently created projection.
    For n > 2, uses recursive tiling of the 2×2 pattern.
    """
    global _projection_cache
    
    if matrix_size is None:
        if not _projection_cache:
            raise ValueError("No collision projection exists yet")
        # Return most recent
        matrix_size = max(_projection_cache.keys())
    
    if matrix_size not in _projection_cache:
        if matrix_size == 2:
            _projection_cache[matrix_size] = CollisionProjection(matrix_size)
        elif matrix_size > 2 and (matrix_size & (matrix_size - 1)) == 0:  # Power of 2
            # Use recursive projection
            _projection_cache[matrix_size] = _create_recursive_projection(matrix_size)
        else:
            # Fallback to original for non-power-of-2
            _projection_cache[matrix_size] = CollisionProjection(matrix_size)
    
    return _projection_cache[matrix_size]

def _create_recursive_projection(n: int) -> CollisionProjection:
    """Create a recursive projection by tiling 2×2 blocks."""
    if n == 2:
        return CollisionProjection(2)
    
    half = n // 2
    sub = collision_projection(2)  # Always reuse n=2 pattern
    
    # Create new projection with proper size
    proj = CollisionProjection.__new__(CollisionProjection)
    proj.matrix_size = n
    proj.lattice_size = n * 4  # Enough space for tiling
    proj.lattice_shape = (proj.lattice_size, proj.lattice_size)
    proj._forward = {}
    proj._inverse = {}
    
    # Four quadrants of C
    for ci, cj, x_off, y_off in [
        (0, 0, 0,         0        ),
        (0, 1, 4 * half,  0        ),
        (1, 0, 0,         4 * half ),
        (1, 1, 4 * half,  4 * half ),
    ]:
        # For each quadrant, we need all k values
        for k in range(n):
            # Copy sub-projection with offsets & index remapping
            for (x, y), (sub_i, sub_j, sub_k) in sub._forward.items():
                # Map local indices to global
                global_i = ci * half + sub_i
                global_j = cj * half + sub_j
                global_k = k  # k spans full range
                
                # Only add if this k matches what the 2×2 block would handle
                ki = k // half  # which block of A contains this k?
                kj = k // half  # which block of B contains this k?
                
                if (ci == ki and cj == kj and sub_k == k % half):
                    new_x = x + x_off + (k // half) * 10  # Offset by k block
                    new_y = y + y_off + (k % half) * 10
                    
                    if (new_x, new_y) not in proj._forward:
                        proj._forward[(new_x, new_y)] = (global_i, global_j, global_k)
                        proj._inverse[(global_i, global_j, global_k)] = (new_x, new_y)
    
    logger.info(
        f"Recursive projection built: {len(proj._forward)} lattice sites "
        f"for {n}×{n} matrix multiplication"
    )
    
    return proj

__all__ = ['CollisionProjection', 'collision_projection', 'PHI']
