"""
ALBERT Phase Encoding Module
Converts spacetime curvature into phase and amplitude fields for ψ-mesh integration
"""

import numpy as np
import sympy as sp
from typing import Dict, Any, Union, List, Tuple
import logging

logger = logging.getLogger(__name__)


def encode_curvature_to_phase(
    kretschmann_scalar: Union[sp.Expr, float],
    coords: List[str],
    region_sample: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Convert Kretschmann curvature scalar into phase and amplitude modulations
    
    Args:
        kretschmann_scalar: Symbolic or numeric Kretschmann scalar K = R^μνρσ R_μνρσ
        coords: List of coordinate names ['t', 'r', 'theta', 'phi']
        region_sample: Dictionary with coordinate arrays to evaluate over
                      e.g., {"r": np.linspace(1.9, 3.0, 100), 
                             "theta": np.array([np.pi/2])}
    
    Returns:
        Dictionary containing:
            - "psi_phase": Phase shift field (curvature-driven phase twists)
            - "psi_amplitude": Amplitude modulation field (memory expansion/compression)
    """
    
    # Extract coordinate symbols
    coord_symbols = sp.symbols(coords)
    coord_dict = {coord: sym for coord, sym in zip(coords, coord_symbols)}
    
    # Create evaluation grid
    grid_shape = [len(region_sample.get(coord, [0])) for coord in coords]
    
    # Initialize output arrays
    psi_phase = np.zeros(grid_shape)
    psi_amplitude = np.ones(grid_shape)  # Start with unit amplitude
    
    # If Kretschmann is symbolic, create a lambdified function
    if isinstance(kretschmann_scalar, sp.Expr):
        try:
            # Lambdify for numerical evaluation
            K_func = sp.lambdify(coord_symbols, kretschmann_scalar, 'numpy')
        except Exception as e:
            logger.warning(f"Failed to lambdify Kretschmann scalar: {e}")
            # Fall back to constant
            K_func = lambda *args: float(kretschmann_scalar.subs({sym: 0 for sym in coord_symbols}))
    else:
        # Constant scalar
        K_func = lambda *args: float(kretschmann_scalar)
    
    # Create meshgrid for evaluation
    mesh_coords = {}
    for coord in coords:
        if coord in region_sample:
            mesh_coords[coord] = region_sample[coord]
        else:
            # Default values for unspecified coordinates
            if coord == 't':
                mesh_coords[coord] = np.array([0])
            elif coord == 'phi':
                mesh_coords[coord] = np.array([0])
            else:
                mesh_coords[coord] = np.array([1.0])
    
    # Create full meshgrid
    mesh_arrays = np.meshgrid(*[mesh_coords[coord] for coord in coords], indexing='ij')
    
    # Evaluate Kretschmann scalar over the grid
    try:
        K_values = K_func(*mesh_arrays)
    except Exception as e:
        logger.error(f"Error evaluating Kretschmann scalar: {e}")
        K_values = np.zeros_like(mesh_arrays[0])
    
    # Phase encoding: curvature creates phase twists
    # High curvature = stronger phase modulation
    # Normalize by Planck scale curvature K_p ~ 1
    K_planck = 1.0  # In geometric units
    
    # Phase shift proportional to log of curvature (captures wide dynamic range)
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    psi_phase = np.angle(np.exp(1j * np.log(np.abs(K_values) + epsilon) / K_planck))
    
    # Amplitude modulation: curvature affects memory density
    # High curvature = compression (smaller amplitude)
    # Low curvature = expansion (larger amplitude)
    # Use sigmoid-like function for smooth transition
    K_critical = 0.1  # Critical curvature scale
    psi_amplitude = 1.0 / (1.0 + (K_values / K_critical) ** 2)
    
    # Special handling for singularities and horizons
    # Detect extreme curvature regions
    extreme_mask = np.abs(K_values) > 100 * K_critical
    
    # Near singularities: phase becomes chaotic, amplitude collapses
    if np.any(extreme_mask):
        # Add phase noise near singularities
        phase_noise = np.random.uniform(-np.pi, np.pi, size=psi_phase.shape)
        psi_phase = np.where(extreme_mask, phase_noise, psi_phase)
        
        # Amplitude approaches zero (memory collapse)
        psi_amplitude = np.where(extreme_mask, 1e-3, psi_amplitude)
        
        logger.warning(f"Detected {np.sum(extreme_mask)} points with extreme curvature")
    
    # Ensure phase is in [-π, π]
    psi_phase = np.angle(np.exp(1j * psi_phase))
    
    # Log diagnostics
    logger.info(f"Phase modulation computed over grid shape: {grid_shape}")
    logger.info(f"Curvature range: [{np.min(K_values):.2e}, {np.max(K_values):.2e}]")
    logger.info(f"Phase range: [{np.min(psi_phase):.2f}, {np.max(psi_phase):.2f}]")
    logger.info(f"Amplitude range: [{np.min(psi_amplitude):.2f}, {np.max(psi_amplitude):.2f}]")
    
    return {
        "psi_phase": psi_phase,
        "psi_amplitude": psi_amplitude,
        "curvature_values": K_values,  # Include raw curvature for reference
        "grid_shape": grid_shape,
        "coordinates": mesh_coords
    }


def phase_twist_geodesic(
    psi_phase_field: np.ndarray,
    geodesic_path: List[Tuple[float, ...]],
    coords: List[str]
) -> float:
    """
    Compute accumulated phase twist along a geodesic path
    
    Args:
        psi_phase_field: Phase field from encode_curvature_to_phase
        geodesic_path: List of coordinate tuples along the path
        coords: Coordinate names for interpretation
        
    Returns:
        Total phase accumulation along path
    """
    total_phase = 0.0
    
    for i in range(len(geodesic_path) - 1):
        # Interpolate phase at each point
        # (Simplified - assumes grid alignment)
        point_phase = psi_phase_field[0]  # Placeholder
        total_phase += point_phase
    
    return total_phase


def amplitude_lensing_factor(
    psi_amplitude_field: np.ndarray,
    source_pos: Tuple[float, ...],
    observer_pos: Tuple[float, ...]
) -> float:
    """
    Compute gravitational lensing amplification factor
    
    Args:
        psi_amplitude_field: Amplitude field from encode_curvature_to_phase
        source_pos: Source position coordinates
        observer_pos: Observer position coordinates
        
    Returns:
        Lensing amplification factor
    """
    # Simplified: return amplitude at midpoint
    # Full implementation would integrate along null geodesic
    return 1.0  # Placeholder


def inject_phase_into_concept_mesh(
    concept_id: str,
    phase_value: float,
    amplitude_value: float
) -> None:
    """
    Inject phase modulation into concept mesh node
    
    Args:
        concept_id: Identifier for concept node
        phase_value: Phase shift to apply
        amplitude_value: Amplitude scaling factor
    """
    logger.info(f"Injecting phase {phase_value:.2f}, amplitude {amplitude_value:.2f} "
                f"into concept {concept_id}")
    # This would interface with the actual ψ-mesh implementation
