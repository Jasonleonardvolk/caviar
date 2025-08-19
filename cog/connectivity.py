"""
Connectivity matrix utilities for cognitive architecture.
"""
import numpy as np

__all__ = ["create_connectivity_matrix"]

_DEFAULT_REGIONS = (
    "perception",
    "memory",
    "reasoning",
    "action",
)

def create_connectivity_matrix(nodes=None) -> np.ndarray:
    """
    Returns a weighted adjacency matrix (NxN).
    For now: fully-connected with decay by distance in list order.
    
    Args:
        nodes: List/tuple of node names. Defaults to standard cognitive regions.
        
    Returns:
        NxN numpy array representing connectivity weights
    """
    if nodes is None:
        nodes = _DEFAULT_REGIONS
        
    n = len(nodes)
    W = np.zeros((n, n), dtype=float)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Weight decreases with distance in the list
            W[i, j] = 1.0 / (1 + abs(i - j))
            
    return W
