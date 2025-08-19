#!/usr/bin/env python3
"""
Side mode module for χ-reduced interferometric lattices.
Implements analytic Σ(ω) = Σ_cont + Σ_poles and returns NxN matrix for BdG solver.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import logging
from functools import lru_cache

try:
    import cupy as xp
    CUPY_AVAILABLE = True
except ImportError:
    import numpy as xp
    CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Constants
CONTINUUM_CUTOFF = 1e-12
CACHE_SIZE = 128


def self_energy(
    chi: float, 
    pole_list: Optional[List[Tuple[float, float]]] = None,
    n: int = 100,
    omega: Optional[Union[float, np.ndarray]] = None,
    gap: float = 1.0,
    gamma_mv: float = 1.0,
    use_gpu: bool = False
) -> np.ndarray:
    """
    Calculate the total self-energy matrix Σ(ω) = Σ_cont(ω) + Σ_poles(ω).
    
    Args:
        chi: Geometric factor (0 ≤ χ ≤ 1)
        pole_list: List of (omega_pole, gamma_pole) tuples
        n: Size of the matrix (number of lattice sites)
        omega: Frequency (scalar or array). If None, returns operator form.
        gap: Superconducting gap Δ
        gamma_mv: Multi-valued hybridization strength
        use_gpu: Whether to use GPU acceleration if available
        
    Returns:
        Σ: Self-energy matrix of shape (n, n) or (len(omega), n, n)
    """
    # Select backend
    if use_gpu and CUPY_AVAILABLE:
        backend = xp
    else:
        backend = np
        
    # Initialize self-energy
    if omega is None:
        # Return operator form for eigenvalue problems
        sigma = backend.zeros((n, n), dtype=complex)
    elif np.isscalar(omega):
        sigma = backend.zeros((n, n), dtype=complex)
    else:
        omega = backend.asarray(omega)
        sigma = backend.zeros((len(omega), n, n), dtype=complex)
    
    # Add continuous contribution
    sigma_cont = continuum_self_energy(chi, n, omega, gap, gamma_mv, backend)
    sigma += sigma_cont
    
    # Add pole contributions
    if pole_list:
        sigma_poles = pole_self_energy(pole_list, n, omega, backend)
        sigma += sigma_poles
    
    # Convert back to numpy if needed
    if use_gpu and CUPY_AVAILABLE:
        sigma = xp.asnumpy(sigma)
        
    return sigma


@lru_cache(maxsize=CACHE_SIZE)
def continuum_self_energy(
    chi: float,
    n: int,
    omega: Optional[Union[float, Tuple[float, ...]]] = None,
    gap: float = 1.0,
    gamma_mv: float = 1.0,
    backend = np
) -> np.ndarray:
    """
    Calculate the continuous part of the self-energy.
    
    From Eq. (61) of the paper:
    D_cont(ω) = Γ^mv sgn(ω) √(ω² - Δ²) / s(ω) * [[ω, -Δχ], [-Δχ, ω]]
    
    The self-energy is the Hilbert transform of the hybridization function.
    """
    # Convert tuple back to array for caching
    if isinstance(omega, tuple):
        omega = backend.array(omega)
        
    if omega is None:
        # Operator form - use effective approximation
        sigma_cont = backend.zeros((n, n), dtype=complex)
        # Diagonal dominates for large systems
        sigma_cont[np.diag_indices(n)] = -1j * gamma_mv * chi * gap
        return sigma_cont
    
    # Ensure omega is array-like
    omega_arr = backend.atleast_1d(omega)
    scalar_input = omega_arr.shape[0] == 1
    
    # Initialize output
    if scalar_input:
        sigma_cont = backend.zeros((n, n), dtype=complex)
    else:
        sigma_cont = backend.zeros((len(omega_arr), n, n), dtype=complex)
    
    # Calculate for each frequency
    for i, w in enumerate(omega_arr):
        if backend.abs(w) <= gap + CONTINUUM_CUTOFF:
            # Inside gap - use analytical continuation
            sigma_w = _inside_gap_continuation(w, chi, gap, gamma_mv, n, backend)
        else:
            # Outside gap - use direct formula
            sigma_w = _outside_gap_formula(w, chi, gap, gamma_mv, n, backend)
        
        if scalar_input:
            sigma_cont = sigma_w
        else:
            sigma_cont[i] = sigma_w
    
    return sigma_cont


def _inside_gap_continuation(w, chi, gap, gamma_mv, n, backend):
    """Analytical continuation of self-energy inside the gap."""
    # Use Kramers-Kronig to get real part
    # For |ω| < Δ, we have branch cut contribution
    eta = 1e-10  # Small imaginary part for causality
    
    # Leading order approximation for χ-reduced system
    sigma = backend.zeros((n, n), dtype=complex)
    
    # Diagonal part
    diagonal = -gamma_mv * chi * w / backend.sqrt(gap**2 - w**2 + eta**2)
    sigma[backend.diag_indices(n)] = diagonal
    
    # Off-diagonal coupling from χ
    if chi > CONTINUUM_CUTOFF:
        # Nearest-neighbor approximation
        off_diag = -gamma_mv * chi * gap / (2 * backend.sqrt(gap**2 - w**2 + eta**2))
        for i in range(n-1):
            sigma[i, i+1] = off_diag
            sigma[i+1, i] = off_diag
    
    return sigma


def _outside_gap_formula(w, chi, gap, gamma_mv, n, backend):
    """Direct formula for self-energy outside the gap."""
    # From the paper's hybridization function
    sigma = backend.zeros((n, n), dtype=complex)
    
    # s(ω) function that appears in denominator
    # Simplified for two-terminal case
    s_omega = w**2 - gap**2 * (1 - chi**2)
    
    if backend.abs(s_omega) < CONTINUUM_CUTOFF:
        # Near divergence - regularize
        s_omega = backend.sign(s_omega) * CONTINUUM_CUTOFF
    
    # Prefactor
    prefactor = gamma_mv * backend.sign(w) * backend.sqrt(w**2 - gap**2) / s_omega
    
    # Diagonal elements
    sigma[backend.diag_indices(n)] = prefactor * w
    
    # Off-diagonal from χ
    if chi > CONTINUUM_CUTOFF:
        off_diag = -prefactor * gap * chi
        for i in range(n-1):
            sigma[i, i+1] = off_diag
            sigma[i+1, i] = off_diag
    
    return sigma


def pole_self_energy(
    pole_list: List[Tuple[float, float]],
    n: int,
    omega: Optional[Union[float, np.ndarray]] = None,
    backend = np
) -> np.ndarray:
    """
    Calculate the pole contribution to self-energy.
    
    Each pole contributes a Lorentzian:
    Σ_pole(ω) = Σ_p Γ_p / (ω - ω_p + iη)
    """
    if omega is None:
        # Operator form - sum of pole projectors
        sigma_poles = backend.zeros((n, n), dtype=complex)
        for omega_p, gamma_p in pole_list:
            # Add projector for each pole
            # This is an approximation - exact form requires eigenvector
            projector = backend.zeros((n, n), dtype=complex)
            # Localized state approximation
            center = n // 2
            projector[center, center] = gamma_p / (1j * 1e-6)  # Regularized
            sigma_poles += projector
        return sigma_poles
    
    # Ensure omega is array-like
    omega_arr = backend.atleast_1d(omega)
    scalar_input = omega_arr.shape[0] == 1
    
    # Initialize output
    if scalar_input:
        sigma_poles = backend.zeros((n, n), dtype=complex)
    else:
        sigma_poles = backend.zeros((len(omega_arr), n, n), dtype=complex)
    
    # Sum over all poles
    for omega_p, gamma_p in pole_list:
        # Lorentzian broadening parameter
        eta = 1e-6
        
        # Calculate contribution for each frequency
        for i, w in enumerate(omega_arr):
            # Lorentzian line shape
            denominator = w - omega_p + 1j * eta
            pole_contrib = gamma_p / denominator
            
            # Create pole state (localized in center for simplicity)
            pole_matrix = backend.zeros((n, n), dtype=complex)
            
            # In reality, this should be the eigenvector of the side-coupled mode
            # For now, use a localized state
            center = n // 2
            width = max(1, n // 20)  # 5% of system size
            
            for j in range(max(0, center - width), min(n, center + width + 1)):
                weight = backend.exp(-(j - center)**2 / (2 * width**2))
                pole_matrix[j, j] = pole_contrib * weight
            
            if scalar_input:
                sigma_poles += pole_matrix
            else:
                sigma_poles[i] += pole_matrix
    
    return sigma_poles


def effective_hamiltonian(
    chi: float,
    pole_list: Optional[List[Tuple[float, float]]] = None,
    n: int = 100,
    gap: float = 1.0,
    gamma_mv: float = 1.0,
    laplacian: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Construct effective Hamiltonian including side-coupled modes.
    
    H_eff = H_0 + Σ(0)
    
    Args:
        chi: Geometric factor
        pole_list: List of poles
        n: System size
        gap: Superconducting gap
        gamma_mv: Hybridization strength
        laplacian: Base Laplacian (if None, use nearest-neighbor)
        
    Returns:
        H_eff: Effective Hamiltonian matrix
    """
    # Base Hamiltonian
    if laplacian is None:
        # Default nearest-neighbor hopping
        H0 = np.zeros((n, n), dtype=complex)
        t = 1.0  # Hopping amplitude
        for i in range(n-1):
            H0[i, i+1] = -t
            H0[i+1, i] = -t
    else:
        H0 = laplacian.copy()
    
    # Add self-energy at zero frequency (static approximation)
    sigma_0 = self_energy(chi, pole_list, n, omega=0.0, gap=gap, gamma_mv=gamma_mv)
    
    H_eff = H0 + sigma_0
    
    return H_eff


def validate_parameters(
    chi: float,
    pole_list: Optional[List[Tuple[float, float]]] = None,
    gap: float = 1.0
) -> bool:
    """
    Validate input parameters are physical.
    
    Returns:
        True if all parameters are valid
    """
    # Check chi range
    if not (0 <= chi <= 1):
        logger.error(f"Invalid chi = {chi}, must be in [0, 1]")
        return False
    
    # Check gap
    if gap <= 0:
        logger.error(f"Invalid gap = {gap}, must be positive")
        return False
    
    # Check poles
    if pole_list:
        for omega_p, gamma_p in pole_list:
            if abs(omega_p) >= gap:
                logger.error(f"Pole at ω = {omega_p} outside gap")
                return False
            if gamma_p < 0:
                logger.error(f"Negative pole strength Γ = {gamma_p}")
                return False
    
    return True


# Performance monitoring
def benchmark_self_energy(n_values: List[int] = [50, 100, 200, 500]):
    """Benchmark self-energy calculation for different system sizes."""
    import time
    
    results = {}
    chi = 0.5
    pole_list = [(0.5, 0.1), (-0.5, 0.1)]
    omega = np.linspace(-2, 2, 100)
    
    for n in n_values:
        # CPU timing
        start = time.time()
        sigma_cpu = self_energy(chi, pole_list, n, omega, use_gpu=False)
        cpu_time = time.time() - start
        
        results[n] = {'cpu': cpu_time}
        
        # GPU timing if available
        if CUPY_AVAILABLE:
            start = time.time()
            sigma_gpu = self_energy(chi, pole_list, n, omega, use_gpu=True)
            gpu_time = time.time() - start
            results[n]['gpu'] = gpu_time
            
            # Verify results match
            if not np.allclose(sigma_cpu, sigma_gpu, rtol=1e-6):
                logger.warning(f"CPU/GPU mismatch for n={n}")
        
        logger.info(f"n={n}: CPU={cpu_time:.3f}s" + 
                   (f", GPU={gpu_time:.3f}s" if CUPY_AVAILABLE else ""))
    
    return results
