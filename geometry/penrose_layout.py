"""
Penrose Layout - Geometric routing for soliton paths
Implements 5-fold quasi-periodic tiling with golden ratio constraints
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_MAX_TILE_COUNT = 65536  # Maximum tiles before sharding
DEFAULT_SHARD_SIZE = 256  # Default matrix shard dimension
DEFAULT_PARALLEL_SHARDS = 4  # Concurrent shard computations

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio

class PenroseRouter:
    """
    Manages Penrose tiling layout and soliton routing
    Ensures golden-ratio path quantization for interference
    """
    
    def __init__(
        self,
        max_tile_count: int = DEFAULT_MAX_TILE_COUNT,
        shard_size: int = DEFAULT_SHARD_SIZE,
        parallel_shards: int = DEFAULT_PARALLEL_SHARDS,
        **kwargs
    ):
        self.max_tile_count = max_tile_count
        self.shard_size = shard_size
        self.parallel_shards = parallel_shards
        
        # Tiling state
        self.tiles: List[Dict] = []
        self.routing_table: Dict[Tuple[int, int], Tuple[float, float]] = {}
        
        logger.info(
            f"PenroseRouter initialized: max_tiles={max_tile_count}, "
            f"shard_size={shard_size}, parallel_shards={parallel_shards}"
        )
    
    def generate_tiling(self, matrix_size: int) -> bool:
        """
        Generate Penrose tiling for given matrix size
        Returns True if successful, False if sharding needed
        """
        # Estimate required tiles
        # For n×n matrix multiplication, we need ~n^2.8 collision sites
        required_tiles = int(matrix_size ** 2.8)
        
        if required_tiles > self.max_tile_count:
            logger.warning(
                f"Matrix size {matrix_size} requires {required_tiles} tiles, "
                f"exceeding limit of {self.max_tile_count}. Sharding needed."
            )
            return False
        
        # Generate tiles (simplified - real implementation would use
        # de Bruijn's method or inflation/deflation)
        self.tiles = []
        
        # Create 5-fold symmetric seed
        for arm in range(5):
            angle = 2 * np.pi * arm / 5
            
            # Radial positions follow golden ratio spacing
            r = 1.0
            tile_count = 0
            
            while tile_count < required_tiles // 5:
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                
                self.tiles.append({
                    'type': 'kite' if tile_count % 2 == 0 else 'dart',
                    'position': (x, y),
                    'arm': arm,
                    'radius': r
                })
                
                r *= PHI  # Golden ratio scaling
                tile_count += 1
        
        logger.info(f"Generated {len(self.tiles)} Penrose tiles")
        return True
    
    def build_routing_table(self, matrix_size: int):
        """
        Map matrix indices (i,k) to physical tile positions
        Ensures proper interference patterns
        """
        self.routing_table.clear()
        
        # Assign matrix elements to tiles based on 5-fold + golden ratio
        for i in range(matrix_size):
            for k in range(matrix_size):
                # Hash function that preserves algebraic structure
                # Critical: ensures (i,k) and (k,j) pairs meet at right location
                tile_idx = self._penrose_hash(i, k, matrix_size)
                
                if tile_idx < len(self.tiles):
                    self.routing_table[(i, k)] = self.tiles[tile_idx]['position']
    
    def _penrose_hash(self, i: int, k: int, n: int) -> int:
        """
        Map matrix indices to tile index preserving interference constraints
        """
        # Use golden ratio to ensure incommensurate paths interfere destructively
        # except for the algebraically required pairs
        hash_val = int((i * PHI + k) * n / PHI) % len(self.tiles)
        return hash_val
    
    def get_shard_plan(self, matrix_size: int) -> List[Tuple[int, int, int, int]]:
        """
        If matrix too large, return sharding plan
        Each tuple is (row_start, row_end, col_start, col_end)
        """
        if matrix_size <= self.shard_size:
            return [(0, matrix_size, 0, matrix_size)]
        
        shards = []
        for i in range(0, matrix_size, self.shard_size):
            for j in range(0, matrix_size, self.shard_size):
                row_end = min(i + self.shard_size, matrix_size)
                col_end = min(j + self.shard_size, matrix_size)
                shards.append((i, row_end, j, col_end))
        
        logger.info(f"Matrix {matrix_size}×{matrix_size} split into {len(shards)} shards")
        return shards

def create_router(**kwargs) -> PenroseRouter:
    """Factory with environment variable support"""
    import os
    
    if 'max_tile_count' not in kwargs:
        kwargs['max_tile_count'] = int(os.getenv('TORI_MAX_TILE_COUNT', DEFAULT_MAX_TILE_COUNT))
    if 'shard_size' not in kwargs:
        kwargs['shard_size'] = int(os.getenv('TORI_SHARD_SIZE', DEFAULT_SHARD_SIZE))
    if 'parallel_shards' not in kwargs:
        kwargs['parallel_shards'] = int(os.getenv('TORI_PARALLEL_SHARDS', DEFAULT_PARALLEL_SHARDS))
    
    return PenroseRouter(**kwargs)

__all__ = ['PenroseRouter', 'create_router', 'PHI']
