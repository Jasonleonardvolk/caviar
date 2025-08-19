"""
SIMD-friendly mixed precision encoding for soliton physics
"""
import numpy as np
from numba import njit, float32, float64, complex64

@njit(fastmath=True, cache=True)
def encode_matrices_f32_simd(A32: np.ndarray, B32: np.ndarray, positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    SIMD-optimized encoding of matrix values to wavefield.
    
    Parameters
    ----------
    A32, B32 : float32 arrays
        Input matrices already cast to float32
    positions : int array of shape (N, 2)
        The (x, y) positions to write to
    indices : int array of shape (N, 3)
        The (i, j, k) indices for each position
        
    Returns
    -------
    field : complex64 array
        3-layer wavefield with encoded values
    """
    n_positions = len(positions)
    if n_positions == 0:
        return np.zeros((1, 1, 3), dtype=np.complex64)
    
    # Find bounds for allocation
    max_x = np.max(positions[:, 0]) + 2
    max_y = np.max(positions[:, 1]) + 2
    
    # Allocate field as complex64 (half the memory of complex128)
    field = np.zeros((max_x, max_y, 3), dtype=np.complex64)
    
    # Vectorized encoding - compiler will SIMD this
    for idx in range(n_positions):
        x = positions[idx, 0]
        y = positions[idx, 1]
        i = indices[idx, 0]
        j = indices[idx, 1]
        k = indices[idx, 2]
        
        # Direct value encoding (sign included)
        val_a = A32[i, k]
        val_b = B32[k, j]
        
        # Layer 0: left operand, Layer 1: right operand
        field[x, y, 0] = val_a
        field[x, y, 1] = val_b
    
    return field


@njit(fastmath=True, cache=True)
def prepare_encoding_data(proj_data):
    """
    Pre-process projection data for vectorized encoding.
    Returns positions and indices as contiguous arrays.
    """
    n = len(proj_data)
    positions = np.empty((n, 2), dtype=np.int32)
    indices = np.empty((n, 3), dtype=np.int32)
    
    for i, ((x, y), (ii, jj, kk)) in enumerate(proj_data):
        positions[i, 0] = x
        positions[i, 1] = y
        indices[i, 0] = ii
        indices[i, 1] = jj
        indices[i, 2] = kk
    
    return positions, indices


def encode_with_mixed_precision(A, B, proj):
    """
    High-level encoding function using mixed precision.
    Inputs are float64, encoding uses float32, accumulation stays float64.
    
    IMPORTANT: The caller must know about the scaling to properly decode!
    The original code in _soliton_block_2x2 expects the scaling to be done
    and scales back by max_amp**2.
    """
    # Prepare projection data
    proj_data = list(proj.inverse_items())
    positions, indices = prepare_encoding_data(proj_data)
    
    # IMPORTANT: We need to match the original encoding's scale behavior
    # The original code scales by 1/max_amp in encoding
    max_amp = max(np.abs(A).max(), np.abs(B).max())
    scale = 1.0 / max_amp if max_amp > 0 else 1.0
    
    # Cast to float32 and apply scale
    A32 = (A * scale).astype(np.float32)
    B32 = (B * scale).astype(np.float32)
    
    # SIMD-optimized encoding
    field_c64 = encode_matrices_f32_simd(A32, B32, positions, indices)
    
    # Return as complex128 for compatibility
    return field_c64.astype(np.complex128)


__all__ = ['encode_matrices_f32_simd', 'prepare_encoding_data', 'encode_with_mixed_precision']
