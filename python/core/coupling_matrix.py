#!/usr/bin/env python3
"""
CouplingMatrix – adaptive K_ij management with Hebbian rules.
"""
from __future__ import annotations
import numpy as np


class CouplingMatrix:
    def __init__(self, n: int, k_init: float = 0.0):
        self.K = np.full((n, n), k_init, dtype=np.float32)
    
    def resize(self, n_new: int, k_init: float = 0.0) -> None:
        if n_new <= self.K.shape[0]:
            return
        
        n_old = self.K.shape[0]
        K_new = np.full((n_new, n_new), k_init, dtype=np.float32)
        K_new[:n_old, :n_old] = self.K
        self.K = K_new
    
    # ---------- Hebbian updates ---------- #
    def strengthen(self, i: int, j: int, delta: float = 0.01) -> None:
        self.K[i, j] += delta
        self.K[j, i] += delta  # symmetric
    
    def decay(self, rate: float = 1e-4) -> None:
        self.K *= (1.0 - rate)
