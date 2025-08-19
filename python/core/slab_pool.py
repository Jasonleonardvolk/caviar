"""
Slab allocator for late materialization of Strassen temporaries
"""
import numpy as np
import threading
import contextlib
import collections

# Thread-safe slab storage
_Slabs = collections.defaultdict(list)
_Lock = threading.Lock()

@contextlib.contextmanager
def get(shape, dtype=np.float64):
    """
    Get a temporary buffer from the slab pool.
    Buffer is automatically returned when context exits.
    
    Parameters
    ----------
    shape : tuple
        Shape of the array needed
    dtype : numpy dtype
        Data type (default: float64)
        
    Yields
    ------
    array : np.ndarray
        Temporary array from pool or newly allocated
    """
    key = (shape, dtype)
    
    # Try to get from pool
    with _Lock:
        try:
            arr = _Slabs[key].pop()
        except IndexError:
            # Allocate new if pool is empty
            arr = np.empty(shape, dtype=dtype)
    
    try:
        yield arr
    finally:
        # Return to pool
        with _Lock:
            _Slabs[key].append(arr)


def clear_pool():
    """Clear all cached slabs (for memory management)"""
    with _Lock:
        _Slabs.clear()


def pool_stats():
    """Get statistics about the slab pool"""
    with _Lock:
        total_slabs = sum(len(slabs) for slabs in _Slabs.values())
        total_memory = 0
        for (shape, dtype), slabs in _Slabs.items():
            if slabs:
                bytes_per = np.prod(shape) * np.dtype(dtype).itemsize
                total_memory += bytes_per * len(slabs)
        
        return {
            'num_pools': len(_Slabs),
            'total_slabs': total_slabs,
            'memory_mb': total_memory / (1024 * 1024)
        }


# Convenience function for Strassen temporaries
@contextlib.contextmanager
def strassen_temps(n, count=2):
    """
    Get multiple temporary matrices for Strassen operations.
    
    Parameters
    ----------
    n : int
        Matrix size (n x n)
    count : int
        Number of temporaries needed
        
    Yields
    ------
    temps : list of arrays
        List of temporary matrices
    """
    temps = []
    contexts = []
    
    try:
        for _ in range(count):
            ctx = get((n, n), dtype=np.float64)
            arr = ctx.__enter__()
            temps.append(arr)
            contexts.append(ctx)
        
        yield temps
        
    finally:
        # Clean up in reverse order
        for ctx in reversed(contexts):
            ctx.__exit__(None, None, None)


__all__ = ['get', 'clear_pool', 'pool_stats', 'strassen_temps']
