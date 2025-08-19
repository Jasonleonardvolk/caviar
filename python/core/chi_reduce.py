#!/usr/bin/env python3
"""
Chi reduction module for interferometric superconducting lattices.
Extracts geometric factor χ and poles from hybridization self-energy.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import logging
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

# Constants
CHI_SIGN = +1  # Single source of truth for sign convention
DEFAULT_CHI = 0.0
POLE_TOLERANCE = 1e-10
GAP_EDGE_BUFFER = 1e-6


def extract_chi(
    graph_edges: List[Tuple[int, int, complex]], 
    flux: float,
    gamma1: float = 1.0,
    gamma2: float = 1.0,
    t_tilde: float = 1.0
) -> float:
    """
    Extract geometric factor χ from interferometer parameters.
    
    Based on Eq. (59) from the paper:
    χ = γ̃₁[e^i(φ₁+φᵥ₁) + t̃²e^i(φ₂+φᵥ₁+φₜ)] + γ̃₂[e^i(φ₂+φᵥ₂) + t̃²e^i(φ₁+φᵥ₂-φₜ)]
    
    Args:
        graph_edges: List of (node1, node2, hopping) tuples
        flux: Magnetic flux through the loop (in units of flux quantum)
        gamma1, gamma2: Hybridization strengths
        t_tilde: Dimensionless inter-lead hopping
        
    Returns:
        chi: Geometric factor (real after gauge transformation)
    """
    # Extract phases from graph structure
    phi1, phi2, phi_v1, phi_v2, phi_t = _extract_phases(graph_edges, flux)
    
    # Effective hybridization
    gamma_mv = 2 * (1 + t_tilde**2) * (gamma1 + gamma2)
    gamma1_tilde = 2 * gamma1 / gamma_mv
    gamma2_tilde = 2 * gamma2 / gamma_mv
    
    # Complex chi before gauge transformation
    chi_complex = (
        gamma1_tilde * (
            np.exp(1j * (phi1 + phi_v1)) + 
            t_tilde**2 * np.exp(1j * (phi2 + phi_v1 + phi_t))
        ) +
        gamma2_tilde * (
            np.exp(1j * (phi2 + phi_v2)) + 
            t_tilde**2 * np.exp(1j * (phi1 + phi_v2 - phi_t))
        )
    )
    
    # Apply gauge transformation to make chi real
    chi = CHI_SIGN * np.abs(chi_complex)
    
    logger.debug(f"Extracted χ = {chi:.6f} from flux = {flux:.3f}")
    return chi


def extract_poles(
    sigma_omega: np.ndarray,
    omega_grid: np.ndarray,
    gap: float = 1.0
) -> List[Tuple[float, float]]:
    """
    Extract poles from hybridization self-energy Σ(ω).
    
    Poles appear at ±ω_pole where:
    ω_pole = ±Δ√[1 - 4t̃²sin²((φ+φₜ)/2)/(t̃²+1)²]
    
    Args:
        sigma_omega: Self-energy evaluated on frequency grid
        omega_grid: Frequency grid points
        gap: Superconducting gap Δ
        
    Returns:
        pole_list: List of (omega_pole, gamma_pole) tuples
    """
    pole_list = []
    
    # Find poles by detecting singularities
    # Real part has 1/(ω² - ω_pole²) structure
    sigma_real = np.real(sigma_omega)
    
    # Look for peaks in |dΣ/dω|
    d_sigma = np.gradient(sigma_real, omega_grid)
    peaks, properties = find_peaks(np.abs(d_sigma), height=1e3, distance=10)
    
    for peak_idx in peaks:
        omega_peak = omega_grid[peak_idx]
        
        # Only consider poles inside the gap
        if np.abs(omega_peak) < gap - GAP_EDGE_BUFFER:
            # Extract residue by fitting Lorentzian
            gamma_pole = _extract_residue(
                sigma_omega, omega_grid, omega_peak
            )
            
            if gamma_pole > POLE_TOLERANCE:
                pole_list.append((omega_peak, gamma_pole))
                logger.debug(f"Found pole at ω = {omega_peak:.4f}, Γ = {gamma_pole:.4f}")
    
    return pole_list


def _extract_phases(
    graph_edges: List[Tuple[int, int, complex]], 
    flux: float
) -> Tuple[float, float, float, float, float]:
    """Extract BCS and hopping phases from graph structure."""
    # This is a simplified extraction - real implementation would
    # parse the full graph topology
    phi_total = 2 * np.pi * flux
    
    # Default phase distribution (can be overridden by graph analysis)
    phi1 = 0.0
    phi2 = phi_total / 2
    phi_v1 = 0.0
    phi_v2 = 0.0
    phi_t = phi_total / 2
    
    return phi1, phi2, phi_v1, phi_v2, phi_t


def _extract_residue(
    sigma_omega: np.ndarray,
    omega_grid: np.ndarray,
    omega_pole: float,
    fit_width: float = 0.1
) -> float:
    """
    Extract pole residue by fitting Lorentzian near the pole.
    
    Near a pole: Σ(ω) ≈ Γ_pole / (ω - ω_pole)
    """
    # Select fitting window
    mask = np.abs(omega_grid - omega_pole) < fit_width
    omega_fit = omega_grid[mask]
    sigma_fit = sigma_omega[mask]
    
    # Avoid the actual pole point
    mask_safe = np.abs(omega_fit - omega_pole) > 1e-4
    omega_fit = omega_fit[mask_safe]
    sigma_fit = sigma_fit[mask_safe]
    
    # Fit 1/ω form to extract residue
    try:
        # For a simple pole: Im[Σ] ~ Γ/(ω-ω₀)
        weights = 1.0 / np.abs(omega_fit - omega_pole)
        gamma_pole = np.abs(np.mean(np.imag(sigma_fit) * (omega_fit - omega_pole)))
        return gamma_pole
    except:
        logger.warning(f"Failed to extract residue at ω = {omega_pole}")
        return 0.0


def chi_reduce(
    lattice_params: Dict,
    flux: float = 0.0
) -> Tuple[float, List[Tuple[float, float]]]:
    """
    Main entry point for chi reduction.
    
    Args:
        lattice_params: Dictionary with lattice configuration
        flux: Magnetic flux through interferometer loops
        
    Returns:
        (chi, pole_list): Geometric factor and list of poles
    """
    # Extract parameters
    gamma1 = lattice_params.get('gamma1', 1.0)
    gamma2 = lattice_params.get('gamma2', 1.0)
    t_tilde = lattice_params.get('t_tilde', 1.0)
    graph_edges = lattice_params.get('edges', [])
    
    # Calculate chi
    chi = extract_chi(graph_edges, flux, gamma1, gamma2, t_tilde)
    
    # Calculate self-energy if needed for pole extraction
    if 'sigma_omega' in lattice_params:
        omega_grid = lattice_params['omega_grid']
        sigma_omega = lattice_params['sigma_omega']
        gap = lattice_params.get('gap', 1.0)
        pole_list = extract_poles(sigma_omega, omega_grid, gap)
    else:
        # Analytical pole positions for simple cases
        pole_list = _analytical_poles(lattice_params, chi)
    
    return chi, pole_list


def _analytical_poles(params: Dict, chi: float) -> List[Tuple[float, float]]:
    """
    Calculate pole positions analytically for known topologies.
    
    From Eq. (57): ω_pole = ±Δ√[1 - 4t̃²sin²((φ+φₜ)/2)/(t̃²+1)²]
    """
    gap = params.get('gap', 1.0)
    t_tilde = params.get('t_tilde', 1.0)
    phi = params.get('phi', 0.0)
    phi_t = params.get('phi_t', 0.0)
    
    # Pole frequency
    sin_arg = np.sin((phi + phi_t) / 2)
    omega_pole_sq = gap**2 * (1 - 4 * t_tilde**2 * sin_arg**2 / (t_tilde**2 + 1)**2)
    
    if omega_pole_sq > 0:
        omega_pole = np.sqrt(omega_pole_sq)
        
        # Pole strength from Eq. (64)
        gamma_mv = params.get('gamma_mv', 1.0)
        gamma_pole = gamma_mv / (t_tilde**2 + 1)**2 * np.sqrt(gap**2 - omega_pole**2)
        
        return [(omega_pole, gamma_pole), (-omega_pole, gamma_pole)]
    else:
        return []


# Validation functions
def validate_chi_range(chi: float) -> bool:
    """Validate that chi is in physical range [0, 1]."""
    return 0.0 <= chi <= 1.0


def validate_poles(pole_list: List[Tuple[float, float]], gap: float) -> bool:
    """Validate that all poles are within the gap."""
    for omega, gamma in pole_list:
        if np.abs(omega) >= gap or gamma < 0:
            return False
    return True
