"""
ALBERT Core Tensors Module
Provides tensor field operations for general relativity calculations
"""

import sympy as sp
from typing import List, Tuple


class TensorField:
    """
    Represents a tensor field on a manifold
    """
    
    def __init__(self, name: str, rank: Tuple[int, int], coords: List[str]):
        """
        Initialize a tensor field
        
        Args:
            name: Name of the tensor (e.g., "g_Kerr" for Kerr metric)
            rank: Tuple of (contravariant_rank, covariant_rank)
            coords: List of coordinate names
        """
        self.name = name
        self.rank = rank
        self.coords = coords
        dim = len(coords)
        
        # Total rank is sum of contravariant and covariant ranks
        total_rank = sum(rank)
        
        # Initialize tensor components as zero
        self.symbols = sp.MutableDenseNDimArray(
            [0] * (dim ** total_rank), 
            shape=(dim,) * total_rank
        )
    
    def set_component(self, indices: Tuple[int, ...], value):
        """Set a specific component of the tensor"""
        self.symbols[indices] = value
    
    def get_component(self, indices: Tuple[int, ...]):
        """Get a specific component of the tensor"""
        return self.symbols[indices]
    
    def simplify(self):
        """Simplify all tensor components symbolically"""
        simplified_data = [sp.simplify(x) for x in self.symbols]
        self.symbols = sp.MutableDenseNDimArray(
            simplified_data, 
            shape=self.symbols.shape
        )
    
    def __repr__(self):
        return f"TensorField({self.name}, rank={self.rank}, shape={self.symbols.shape})"
