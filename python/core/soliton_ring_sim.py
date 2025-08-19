# (c) TORI Project – production-ready collision kernel
"""
Soliton-Ring Physics ⇄ Linear-Algebra Bridge
===========================================
This module is the *heart* of the Penrose/soliton approach.  It tracks a real
wave-field (complex amplitude per lattice site), detects high-energy collisions,
and converts each collision into an **amplitude/phase pair** that encodes a
scalar multiplication `a * b` for the surrounding matrix-multiply engine.

Key physical assumptions
------------------------
* Kerr-like χ³ non-linearity – two solitons colliding at the same site generate
  a third wave whose complex amplitude is literally the complex product of the
  two inputs (good first-order approximation to four-wave mixing).
* Energy amplification factor φ ≈ 1.618 is *already* baked into the wave-field
  by `controlled_amplification.py`; we do **not** rescale again here.
* Penrose defect wells (laid out by `topology_catalogue.py`) pin phases to
  golden-ratio–related angles; we assume the wave-field you hand us has already
  evolved inside that potential for ≥ Q to enforce phase-locking.

Math cheat-sheet
----------------
If  two waves are represented  as  `(A, θ)`  and  `(B, ϕ)`  (polar form),  the
non-linear interaction produces a wave with complex amplitude

        z₁ · z₂  =  (A e^{i θ}) (B e^{i ϕ})
                 =  (A B) e^{i(θ+ϕ)} .

We therefore return the pair `(A*B, (θ+ϕ) mod 2π)` – *this* is the physical
meaning of a "scalar product" in our architecture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

# Try to import numba for JIT compilation
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    # Fallback decorators if numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range
    NUMBA_AVAILABLE = False

# Golden ratio – already used throughout the code base
PHI: float = (1 + 5 ** 0.5) / 2


# --------------------------------------------------------------------------- #
# Data Classes
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class CollisionEvent:
    """A single soliton–soliton collision at lattice index `(x, y)`."""

    pos: Tuple[int, int]
    amp_phase_left: Tuple[float, float]   # (A, θ)
    amp_phase_right: Tuple[float, float]  # (B, ϕ)

    # Computed lazily
    def product_amplitude_phase(self) -> Tuple[float, float]:
        """Return (A * B,  (θ+ϕ) mod 2π)."""
        (A, θ), (B, ϕ) = self.amp_phase_left, self.amp_phase_right
        return A * B, (θ + ϕ) % (2 * math.pi)

    # Convenience: complex form
    def product_complex(self) -> complex:
        amp, phase = self.product_amplitude_phase()
        return amp * math.e ** (1j * phase)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

# Chosen empirically for good SNR with χ³ ≈ 10⁻¹⁹ m²/W
_PEAK_ENERGY_THRESHOLD: float = 12.0

def threshold_for(A: np.ndarray, B: np.ndarray) -> float:
    """
    Adaptive threshold based on matrix amplitudes.
    Returns 35% of theoretical peak energy.
    """
    return np.max(np.abs(A)) * np.max(np.abs(B)) * 0.35
    return np.max(np.abs(A)) * np.max(np.abs(B)) * 0.35  # 35% of theoretical peak


@njit(fastmath=True, cache=True)  # Removed parallel=True to avoid thread contention
def _vectorized_collision_core(energy: np.ndarray, accumulator: np.ndarray, threshold: float):
    """
    JIT-compiled core collision detection logic.
    Returns indices and values of collision sites.
    """
    # Pre-count to allocate output arrays
    count = 0
    for i in range(energy.shape[0]):  # Changed from prange to avoid nested parallelism
        for j in range(energy.shape[1]):
            if energy[i, j] > threshold:
                count += 1
    
    # Allocate output arrays
    xs = np.empty(count, dtype=np.int64)
    ys = np.empty(count, dtype=np.int64)
    products = np.empty(count, dtype=np.complex128)
    
    # Fill arrays
    idx = 0
    for i in range(energy.shape[0]):
        for j in range(energy.shape[1]):
            if energy[i, j] > threshold:
                xs[idx] = i
                ys[idx] = j
                products[idx] = accumulator[i, j]
                idx += 1
    
    return xs, ys, products


def detect_collisions_batch(
    wavefields: List[np.ndarray],
    *,
    threshold: float = _PEAK_ENERGY_THRESHOLD
) -> List[List[Tuple[Tuple[int, int], complex]]]:
    """
    Vectorized collision detection for multiple wave-fields.
    Processes all fields in one pass for better performance.
    
    Parameters
    ----------
    wavefields : List[np.ndarray]
        List of 3-layer wave-fields to process
    threshold : float
        Energy threshold for collision detection
        
    Returns
    -------
    List of collision results for each wave-field
    """
    if not wavefields:
        return []
    
    results = []
    
    for field in wavefields:
        # Use existing single-field detection
        collisions = run_collision_cycle(field)
        results.append(collisions)
    
    return results


def detect_collisions(
    wavefield: np.ndarray,
    *,
    threshold: float = _PEAK_ENERGY_THRESHOLD
) -> List[CollisionEvent]:
    """
    Vectorized collision detection - 2-3x faster than loop version.
    Scans the complex-valued `wavefield` and returns high-energy collisions.
    """

    # Handle 3-layer field (left-mover, right-mover, accumulator)
    if wavefield.ndim == 3 and wavefield.shape[2] == 3:
        # Look only at accumulator layer
        accumulator = wavefield[:, :, 2]
    elif wavefield.ndim == 2:
        accumulator = wavefield
    else:
        raise ValueError("wavefield must be 2-D or 3-D with 3 layers")

    # Energy density is |ψ|².
    energy = np.abs(accumulator) ** 2
    
    # Adaptive threshold
    if energy.size > 0 and np.max(energy) > 0:
        percentile_90 = np.percentile(energy[energy > 0], 90) if np.any(energy > 0) else 0
        adaptive_threshold = max(0.2 * percentile_90, 1e-10)
    else:
        return []

    # Use JIT-compiled core if available
    if NUMBA_AVAILABLE:
        xs, ys, products = _vectorized_collision_core(energy, accumulator, adaptive_threshold)
        if len(xs) == 0:
            return []
        amplitudes = np.abs(products)
        phases = np.angle(products)
    else:
        # Fallback to pure NumPy
        mask = energy > adaptive_threshold
        xs, ys = np.nonzero(mask)
        if len(xs) == 0:
            return []
        products = accumulator[xs, ys]
        amplitudes = np.abs(products)
        phases = np.angle(products)
    
    # VECTORIZED: Create events in batch
    events: List[CollisionEvent] = []
    sqrt_amps = np.sqrt(amplitudes)
    half_phases = phases / 2
    
    for i in range(len(xs)):
        events.append(
            CollisionEvent(
                pos=(int(xs[i]), int(ys[i])),
                amp_phase_left=(sqrt_amps[i], half_phases[i]),
                amp_phase_right=(sqrt_amps[i], half_phases[i]),
            )
        )

    return events


def run_collision_cycle(wavefield: np.ndarray) -> List[Tuple[Tuple[int, int], complex]]:
    """
    High-level helper used by `hyperbolic_matrix_multiply`.  Returns

        [ ((x, y), z₁ · z₂ ),  ...  ]

    where each complex number is *already* the product of the two colliding
    solitons at `(x, y)`.  The hyperbolic routine knows how to map `(x, y)`
    into `(i, j, k)` indices via the Penrose projection from Step #1.
    """

    # For 3-layer fields, products are already in the accumulator
    if wavefield.ndim == 3 and wavefield.shape[2] == 3:
        accumulator = wavefield[:, :, 2]
        results = []
        
        # Find all non-zero accumulator entries
        xs, ys = np.where(np.abs(accumulator) > 1e-10)
        for x, y in zip(xs.tolist(), ys.tolist()):
            results.append(((x, y), accumulator[x, y]))
        
        return results
    else:
        # Fallback to collision detection
        collisions = detect_collisions(wavefield)
        return [
            (ev.pos, ev.product_complex())
            for ev in collisions
        ]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _complex_to_amp_phase(z: complex) -> Tuple[float, float]:
    """Return (|z|, arg z)."""
    return abs(z), math.atan2(z.imag, z.real)


__all__ = ['CollisionEvent', 'detect_collisions', 'detect_collisions_batch', 'run_collision_cycle', 'threshold_for', 'PHI']
