"""
from contextlib import contextmanager
Field Buffer Pool - Reusable wave-field buffers for soliton physics
Reduces memory allocation overhead in recursive calls
"""
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
from contextlib import contextmanager

class FieldPool:
    """
    Per-depth reusable wave-field buffers.
    Reduces allocation overhead in deep recursion.
    """
    
    def __init__(self):
        # Key: (shape, dtype) -> List of available buffers
        self.buffers: Dict[Tuple, List[np.ndarray]] = defaultdict(list)
        self.allocated_count = 0
        self.reuse_count = 0
    
    def acquire(self, shape: Tuple[int, int, int], dtype=np.complex128) -> np.ndarray:
        """
        Get a buffer of the requested shape, reusing if possible.
        
        Parameters
        ----------
        shape : tuple
            Shape of the required array
        dtype : numpy dtype
            Data type (default: complex128)
            
        Returns
        -------
        np.ndarray
            Zeroed array of requested shape
        """
        key = (shape, dtype)
        
        if self.buffers[key]:
            # Reuse existing buffer
            arr = self.buffers[key].pop()
            arr.fill(0)  # Clear it
            self.reuse_count += 1
            return arr
        else:
            # Allocate new buffer
            self.allocated_count += 1
            return np.zeros(shape, dtype=dtype)
    
    def release(self, arr: np.ndarray) -> None:
        """
        Return a buffer to the pool for reuse.
        
        Parameters
        ----------
        arr : np.ndarray
            Array to return to pool
        """
        if arr is None:
            return
            
        key = (arr.shape, arr.dtype)
        self.buffers[key].append(arr)
    
    def clear(self):
        """Clear all buffers (for memory management)"""
        self.buffers.clear()
    
    def stats(self) -> Dict[str, int]:
        """Get usage statistics"""
        total_buffers = sum(len(bufs) for bufs in self.buffers.values())
        return {
            'allocated': self.allocated_count,
            'reused': self.reuse_count,
            'pooled': total_buffers,
            'reuse_rate': self.reuse_count / max(1, self.allocated_count + self.reuse_count)
        }
    
    @contextmanager
    def borrow(self, shape: Tuple[int, ...], dtype=np.float64):
        """Context manager for borrowing and automatically returning buffers"""
        arr = self.acquire(shape, dtype)
        try:
            yield arr
        finally:
            self.release(arr)

# Global pool instance
_global_pool = FieldPool()

def get_field_pool() -> FieldPool:
    """Get the global field pool"""
    return _global_pool

def reset_field_pool():
    """Reset the global field pool"""
    global _global_pool
    _global_pool = FieldPool()

__all__ = ['FieldPool', 'get_field_pool', 'reset_field_pool']
