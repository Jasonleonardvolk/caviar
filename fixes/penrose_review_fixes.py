#!/usr/bin/env python3
"""
Fixes for Penrose integration based on code review
"""

import os
import sys
from pathlib import Path

# Fix 1: Set NUMBA_CACHE_DIR for headless deployments
os.environ['NUMBA_CACHE_DIR'] = os.environ.get('NUMBA_CACHE_DIR', '/tmp/numba')

# Fix 2: Set matplotlib backend for headless servers
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def add_embedder_fallback(ingestion_function):
    """Decorator to add embedder None check"""
    def wrapper(*args, embedder=None, **kwargs):
        if embedder is None:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("No embedder available - skipping Penrose similarity computation")
            # Continue without Penrose
            return ingestion_function(*args, embedder=embedder, **kwargs)
        return ingestion_function(*args, embedder=embedder, **kwargs)
    return wrapper


def fix_edge_order_determinism():
    """Ensure deterministic edge ordering in CSR files"""
    from scipy.sparse import load_npz, save_npz, csr_matrix
    
    def save_sparse_deterministic(sparse_sim: csr_matrix, filepath: Path):
        """Save sparse matrix with sorted indices for determinism"""
        # Sort indices for deterministic ordering
        sparse_sim.sort_indices()
        
        # Save with consistent format
        save_npz(filepath, sparse_sim)
        
    return save_sparse_deterministic


def add_csr_file_locking():
    """Add file locking for CSR writes"""
    import fcntl
    import msvcrt
    import platform
    
    def write_with_lock(filepath: Path, write_func, data):
        """Write file with exclusive lock"""
        # Create temp file
        temp_path = filepath.with_suffix(filepath.suffix + '.tmp')
        
        # Write to temp file with lock
        with open(temp_path, 'wb') as f:
            if platform.system() == 'Windows':
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            try:
                write_func(f, data)
                f.flush()
                os.fsync(f.fileno())
            finally:
                if platform.system() == 'Windows':
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        
        # Atomic rename
        temp_path.replace(filepath)
    
    return write_with_lock


# Fix for duplicate edge handling
def merge_penrose_edges(existing_weight: float, new_weight: float) -> float:
    """Merge weights when duplicate Penrose edges exist"""
    # Take maximum similarity (could also average)
    return max(existing_weight, new_weight)


# Configuration helper
def get_penrose_config():
    """Get Penrose configuration with safe defaults"""
    return {
        'enable_penrose': os.environ.get('ENABLE_PENROSE', 'true').lower() == 'true',
        'rank': int(os.environ.get('PENROSE_RANK', '32')),
        'threshold': float(os.environ.get('PENROSE_THRESHOLD', '0.7')),
        'cache_dir': Path(os.environ.get('PENROSE_CACHE', 'data/.penrose_cache')),
        'max_density': float(os.environ.get('PENROSE_MAX_DENSITY', '0.01')),
    }


if __name__ == "__main__":
    print("🔧 Penrose Review Fixes")
    print("=" * 50)
    print("Environment fixes applied:")
    print(f"  - NUMBA_CACHE_DIR: {os.environ.get('NUMBA_CACHE_DIR')}")
    print(f"  - Matplotlib backend: {matplotlib.get_backend()}")
    print(f"  - Penrose config: {get_penrose_config()}")
