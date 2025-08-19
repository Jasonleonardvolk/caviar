"""
ALBERT Core Module
Mathematical foundations for general relativity
"""

from .manifold import Manifold
from .tensors import TensorField
from .geodesics import (
    numeric_geodesic_solver,
    christoffel_symbols,
    is_null_geodesic,
    is_timelike_geodesic,
    extract_trajectory
)
from .phase_encode import (
    encode_curvature_to_phase,
    phase_twist_geodesic,
    amplitude_lensing_factor,
    inject_phase_into_concept_mesh
)

__all__ = [
    "Manifold", 
    "TensorField",
    "numeric_geodesic_solver",
    "christoffel_symbols",
    "is_null_geodesic",
    "is_timelike_geodesic",
    "extract_trajectory",
    "encode_curvature_to_phase",
    "phase_twist_geodesic",
    "amplitude_lensing_factor",
    "inject_phase_into_concept_mesh"
]
