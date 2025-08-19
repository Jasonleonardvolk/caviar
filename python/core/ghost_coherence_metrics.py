#!/usr/bin/env python3
"""
Utility functions used by GhostSolitonIntegration for richer phase analytics.
"""
from python.core.oscillator_lattice import get_global_lattice
from typing import Dict


def current_metrics() -> Dict[str, float]:
    """Get current lattice metrics for the Ghost subsystem."""
    lattice = get_global_lattice()
    
    return {
        "coherence": lattice.order_parameter(),
        "entropy": lattice.phase_entropy(),
        "count": float(len(lattice.oscillators)),
    }


def get_phase_distribution(bins: int = 36) -> Dict[str, list]:
    """Get binned phase distribution for visualization."""
    lattice = get_global_lattice()
    
    if not lattice.oscillators:
        return {
            "bins": [],
            "counts": [],
            "edges": []
        }
    
    import numpy as np
    phases = np.array([osc.phase for osc in lattice.oscillators])
    counts, edges = np.histogram(phases, bins=bins, range=(0.0, 2*np.pi))
    
    return {
        "bins": [(edges[i] + edges[i+1])/2 for i in range(len(counts))],
        "counts": counts.tolist(),
        "edges": edges.tolist()
    }


def get_coupling_strength_stats() -> Dict[str, float]:
    """Get statistics about the coupling matrix."""
    lattice = get_global_lattice()
    
    if lattice.K is None or len(lattice.oscillators) == 0:
        return {
            "mean": 0.0,
            "max": 0.0,
            "min": 0.0,
            "nonzero_fraction": 0.0
        }
    
    import numpy as np
    K_flat = lattice.K.flatten()
    nonzero = K_flat[K_flat != 0]
    
    return {
        "mean": float(np.mean(K_flat)),
        "max": float(np.max(K_flat)),
        "min": float(np.min(K_flat)),
        "nonzero_fraction": float(len(nonzero) / len(K_flat)) if len(K_flat) > 0 else 0.0
    }
