# bps_transition_manager.py
# ---------------------------------------------------------
# Coordinator for smooth, stepwise BPS topology transitions
# across oscillator-mapped soliton lattices

import logging
import numpy as np
from typing import Optional
from python.core.bps_topology import bps_topology_transition, compute_topological_charge

logger = logging.getLogger("BPSTransition")

class BPSTopologyTransitionManager:
    def __init__(self, lattice: np.ndarray, memory, oscillators: list):
        self.lattice = lattice
        self.memory = memory
        self.oscillators = oscillators
        self.transition_active = False
        self.tick = 0
        self.duration = 0
        self.L1 = None
        self.L2 = None
        self.Q0 = None

    def begin_interpolation(self, L1, L2, duration_ticks: int = 10):
        self.L1 = L1.copy()
        self.L2 = L2.copy()
        self.duration = duration_ticks
        self.tick = 0
        self.transition_active = True
        self.Q0 = compute_topological_charge(self.memory)
        logger.info(f"[BPS-Transition] Starting Laplacian interpolation: Q = {self.Q0}, duration = {duration_ticks}")

    def advance_tick(self):
        if not self.transition_active:
            return False

        alpha = min(self.tick / self.duration, 1.0)
        L_interp = (1 - alpha) * self.L1 + alpha * self.L2
        self.lattice[:] = L_interp[:]
        logger.debug(f"[BPS-Transition] Tick {self.tick} - alpha={alpha:.2f}")
        self.tick += 1

        if self.tick >= self.duration:
            self.finalize()
        return True

    def finalize(self):
        logger.info("[BPS-Transition] Finalizing BPS topology transition...")
        bps_topology_transition(self.L1, self.L2, self.memory, energy_bundle=None)
        self.transition_active = False
        logger.info("[BPS-Transition] Transition complete")

    def cancel(self):
        logger.warning("[BPS-Transition] Transition cancelled")
        self.transition_active = False
        self.tick = 0

    def is_active(self):
        return self.transition_active
