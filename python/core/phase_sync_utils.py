# phase_sync_utils.py
# -------------------------------------------------------------
# Helper utilities for BPS oscillator synchronization, profiling,
# and soliton-aligned phase shaping

import numpy as np
from typing import List
from python.core.bps_oscillator import BPSOscillator

def phase_gradient_profile(center: int, width: int, charge: int, length: int) -> List[float]:
    """Returns phase profile across oscillator array for a kink or vortex."""
    return [charge * np.pi * ((i - center) / width) for i in range(length)]

def entrain_oscillators_to_profile(oscillators: List[BPSOscillator], profile: List[float], gain: float = 1.0):
    for osc, target_phase in zip(oscillators, profile):
        osc.lock_to_phase(target_phase, gain=gain)

def diagnose_locking_error(oscillators: List[BPSOscillator], profile: List[float]) -> float:
    error = [abs(o.theta - p) for o, p in zip(oscillators, profile)]
    return float(sum(error)) / len(error)
