"""
ALBERT API Module
High-level interface for GR computations
"""

from .interface import (
    init_metric, 
    get_metric, 
    describe, 
    trace_geodesic,
    curvature_summary,
    evaluate_penrose_energy_condition,
    check_incomplete_geodesic,
    mark_trapped_surface,
    mark_incomplete_path,
    generate_phase_modulation,
    push_phase_to_ψmesh
)

__all__ = [
    "init_metric", 
    "get_metric", 
    "describe", 
    "trace_geodesic",
    "curvature_summary",
    "evaluate_penrose_energy_condition",
    "check_incomplete_geodesic",
    "mark_trapped_surface",
    "mark_incomplete_path",
    "generate_phase_modulation",
    "push_phase_to_ψmesh"
]
