from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
### File: {PROJECT_ROOT}\albert\api\interface.py

from typing import Optional, Dict, Any, List, Tuple
from albert.metrics.kerr import kerr_metric
from albert.core.manifold import Manifold
from albert.core.tensors import TensorField
from albert.core.geodesics import numeric_geodesic_solver
from albert.core.curvature import compute_riemann_tensor, compute_ricci_tensor, compute_kretschmann_scalar
from albert.core.penrose import (
    energy_condition_satisfied, null_vector_template, is_geodesic_incomplete,
    convergence_scalars, inject_trapped_surface_flag_into_ψmesh,
    inject_incompleteness_flag_into_ψmesh
)
from albert.core.tetrads import canonical_kerr_tetrad
from albert.core.phase_encode import encode_curvature_to_phase
from python.core import ConceptMesh

# Instantiate production-grade ψMesh (singleton)
ψMesh = ConceptMesh.instance()

manifold: Optional[Manifold] = None
metric_tensor: Optional[TensorField] = None


def init_metric(metric_name: str = "kerr", params: Dict[str, Any] = {}) -> TensorField:
    global manifold, metric_tensor
    coords = ['t', 'r', 'theta', 'phi']
    manifold = Manifold(name=metric_name, dimension=4, coordinates=coords)
    if metric_name.lower() == "kerr":
        metric_tensor = kerr_metric(coords, **params)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")
    return metric_tensor


def get_metric() -> Optional[TensorField]:
    return metric_tensor


def describe() -> str:
    if metric_tensor is None:
        return "No metric initialized"
    return str(metric_tensor)


def trace_geodesic(initial_position: List[float], initial_velocity: List[float], lam_span: Tuple[float, float] = (0, 10), steps: int = 500):
    if metric_tensor is None:
        raise RuntimeError("Metric not initialized. Call init_metric first.")
    return numeric_geodesic_solver(metric_tensor, initial_position, initial_velocity, lam_span, steps)


def curvature_summary():
    if metric_tensor is None:
        return "No metric initialized."
    riemann = compute_riemann_tensor(metric_tensor)
    ricci = compute_ricci_tensor(riemann)
    kretsch = compute_kretschmann_scalar(riemann)
    return {
        "riemann": riemann,
        "ricci": ricci,
        "kretschmann": kretsch
    }


def evaluate_penrose_energy_condition():
    if metric_tensor is None:
        raise RuntimeError("Metric not initialized. Call init_metric first.")
    ricci = compute_ricci_tensor(compute_riemann_tensor(metric_tensor))
    k_vec = null_vector_template(metric_tensor.coords)
    return energy_condition_satisfied(ricci, k_vec)


def check_incomplete_geodesic(lam_end: float, lam_max: float = 100):
    return is_geodesic_incomplete(lam_end, lam_max)


def mark_trapped_surface(surface_id: str, r_val: float, theta_val: float):
    tetrad = canonical_kerr_tetrad(metric_tensor.coords, r_val, theta_val)
    surface_point = {"r": r_val, "theta": theta_val}
    ρ, ρp = convergence_scalars(metric_tensor, surface_point, tetrad)
    inject_trapped_surface_flag_into_ψmesh(surface_id, ρ, ρp)
    return ρ, ρp


def mark_incomplete_path(path_id: str, lam_end: float):
    inject_incompleteness_flag_into_ψmesh(path_id, lam_end)


# Production-level: curvature-driven ψMesh modulation

def generate_phase_modulation(region_sample: dict[str, Any]) -> dict[str, Any]:
    kretsch = curvature_summary()['kretschmann']
    return encode_curvature_to_phase(kretsch, metric_tensor.coords, region_sample)


def push_phase_to_ψmesh(surface_id: str, region_sample: dict[str, Any]):
    modulation = generate_phase_modulation(region_sample)
    coords = {k: region_sample[k] for k in region_sample}
    ψMesh.inject_psi_fields(
        concept_id=surface_id,
        psi_phase=modulation['psi_phase'],
        psi_amplitude=modulation['psi_amplitude'],
        coordinates=coords,
        origin="Kretschmann_scalar",
        persistence_mode="persistent"
    )
    print(f"🧠 ψMesh ⟵ Phase injected for region '{surface_id}' (shape {modulation['psi_phase'].shape})")
