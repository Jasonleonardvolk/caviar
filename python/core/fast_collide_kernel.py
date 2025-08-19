# fast_collide_kernel.py
import numpy as np
from numba import njit
from numpy.typing import NDArray

@njit(parallel=False, fastmath=True, cache=True)
def encode32_collide(A: NDArray[np.float64],
                     B: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Fused encode–collide–decode for one 2×2 block.
    Works on float32 inputs (cast upstream) but accumulates in float64.
    Equivalent to 8 scalar FMAs (naive 2×2 mult).
    """
    a00, a01 = A[0, 0], A[0, 1]
    a10, a11 = A[1, 0], A[1, 1]
    b00, b01 = B[0, 0], B[0, 1]
    b10, b11 = B[1, 0], B[1, 1]

    # "Collisions" → plain fused multiplies
    c00 = a00 * b00 + a01 * b10
    c01 = a00 * b01 + a01 * b11
    c10 = a10 * b00 + a11 * b10
    c11 = a10 * b01 + a11 * b11

    out = np.empty((2, 2), dtype=np.float64)
    out[0, 0], out[0, 1] = c00, c01
    out[1, 0], out[1, 1] = c10, c11
    return out
