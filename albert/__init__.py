"""
ALBERT - Advanced Lorentzian/Black-hole Einstein Relativistic Tensor system
A framework for general relativity computations in TORI
"""

from albert.api.interface import (
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
from albert.core.manifold import Manifold
from albert.core.tensors import TensorField

__version__ = "0.1.0"

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
    "push_phase_to_ψmesh",
    "Manifold",
    "TensorField"
]
