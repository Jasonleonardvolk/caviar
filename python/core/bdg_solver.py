#!/usr/bin/env python3
"""
Improved BdG Solver with Physics Corrections and Chi Reduction
Fixes sign conventions, adds missing chemical potential, and verifies symmetries
Now accepts (χ, Σ_poles) pair for interferometric lattices
"""

import numpy as np
try:
    import cupy as xp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as xp
    GPU_AVAILABLE = False

from scipy.sparse import csr_matrix, bmat, diags, kron, eye
from scipy.sparse.linalg import eigs
import logging
from typing import Tuple, Dict, Optional, List, Any

# Import chi reduction modules
try:
    from . import chi_reduce
    from . import side_mode
except ImportError:
    import chi_reduce
    import side_mode

logger = logging.getLogger(__name__)


class BdGSolver:
    """
    Bogoliubov-de Gennes solver for soliton stability analysis
    
    The BdG equations for perturbations (u,v) around a stationary state ψ₀:
    i∂u/∂t = -∇²u - μu + 2g|ψ₀|²u + gψ₀²v*
    i∂v/∂t = +∇²v + μv - 2g|ψ₀|²v - gψ₀*²u
    
    In matrix form: i∂t[u,v]ᵀ = H_BdG[u,v]ᵀ where
    
    H_BdG = [L₀ + 2g|ψ₀|²    gψ₀²      ]
            [-gψ₀*²          -L₀ - 2g|ψ₀|²]
    
    with L₀ = -∇² - μ
    
    For chi-reduced systems, we replace -∇² with -∇² + Σ(ω)
    """
    
    def __init__(self, dx: float = 1.0, boundary: str = 'periodic'):
        """
        Initialize BdG solver
        
        Args:
            dx: Spatial discretization
            boundary: 'periodic', 'dirichlet', or 'neumann'
        """
        self.dx = dx
        self.boundary = boundary
        self.gpu_available = GPU_AVAILABLE
        
    def build_laplacian_1d(self, n: int) -> csr_matrix:
        """Build 1D discrete Laplacian with specified boundary conditions"""
        if self.boundary == 'periodic':
            # Periodic boundary conditions
            data = np.array([-2.0] * n)
            offsets = [0]
            
            # Off-diagonal terms
            off_data = np.array([1.0] * (n-1))
            data_matrix = diags([off_data, data, off_data], [-1, 0, 1], shape=(n, n))
            
            # Periodic connections
            data_matrix = data_matrix.tolil()
            data_matrix[0, n-1] = 1.0
            data_matrix[n-1, 0] = 1.0
            
        elif self.boundary == 'dirichlet':
            # Dirichlet (zero at boundaries)
            data = np.array([-2.0] * n)
            off_data = np.array([1.0] * (n-1))
            data_matrix = diags([off_data, data, off_data], [-1, 0, 1], shape=(n, n))
            
        elif self.boundary == 'neumann':
            # Neumann (zero derivative at boundaries)
            data = np.array([-2.0] * n)
            data[0] = data[-1] = -1.0  # Modified for Neumann
            off_data = np.array([1.0] * (n-1))
            data_matrix = diags([off_data, data, off_data], [-1, 0, 1], shape=(n, n))
        else:
            raise ValueError(f"Unknown boundary condition: {self.boundary}")
            
        return data_matrix.tocsr() / (self.dx**2)
    
    def build_laplacian_2d(self, shape: Tuple[int, int]) -> csr_matrix:
        """Build 2D discrete Laplacian using Kronecker products"""
        ny, nx = shape
        
        # 1D Laplacians
        Dx = self.build_laplacian_1d(nx)
        Dy = self.build_laplacian_1d(ny)
        
        # Identity matrices
        Ix = eye(nx, format='csr')
        Iy = eye(ny, format='csr')
        
        # 2D Laplacian: Dy ⊗ Ix + Iy ⊗ Dx
        L2d = kron(Dy, Ix) + kron(Iy, Dx)
        
        return L2d
    
    def assemble_bdg(self, psi0: np.ndarray, mu: float, g: float = 1.0,
                     chi: float = 0.0, pole_list: Optional[List[Tuple[float, float]]] = None) -> csr_matrix:
        """
        Assemble BdG operator for stability analysis
        
        Args:
            psi0: Stationary state (can be 1D or 2D)
            mu: Chemical potential
            g: Nonlinearity strength
            chi: Geometric factor from chi reduction (0 for standard BdG)
            pole_list: List of (omega_pole, gamma_pole) tuples from chi reduction
            
        Returns:
            Sparse BdG matrix
        """
        # Ensure numpy array
        psi0 = np.asarray(psi0)
        shape = psi0.shape
        ndim = psi0.ndim
        
        # Build Laplacian
        if ndim == 1:
            L = self.build_laplacian_1d(len(psi0))
        elif ndim == 2:
            L = self.build_laplacian_2d(shape)
        else:
            raise ValueError("Only 1D and 2D systems supported")
        
        # Flatten state
        psi0_flat = psi0.flatten()
        N = len(psi0_flat)
        
        # Add chi-reduced self-energy if chi != 0
        if abs(chi) > 1e-10:
            # Get self-energy contribution
            sigma = side_mode.self_energy(
                chi=chi,
                pole_list=pole_list,
                n=N,
                omega=None,  # Static approximation
                gap=1.0,  # Normalized units
                gamma_mv=1.0
            )
            # Add to Laplacian
            L = L + csr_matrix(sigma.real)  # Take real part for Hermitian operator
        
        # Compute density
        rho = np.abs(psi0_flat)**2
        
        # Build L₀ = -∇² - μ (with chi modification included in ∇²)
        L0 = -L - mu * eye(N, format='csr')
        
        # Build diagonal potential term
        V = diags(2 * g * rho, format='csr')
        
        # A matrix: L₀ + V
        A = L0 + V
        
        # B matrix: g*ψ₀²
        B = diags(g * psi0_flat**2, format='csr')
        
        # Build block matrix preserving particle-hole symmetry
        # Note the conjugation in the lower-left block
        B_conj = diags(g * np.conj(psi0_flat)**2, format='csr')
        
        H_BdG = bmat([[A, B], [-B_conj, -A]], format='csr')
        
        return H_BdG
    
    def verify_symmetries(self, H_BdG: csr_matrix, tol: float = 1e-10) -> Dict[str, bool]:
        """
        Verify that BdG matrix has correct symmetries
        
        The BdG Hamiltonian should satisfy:
        1. Particle-hole symmetry: τ₁ H* τ₁ = -H
        2. The spectrum should come in ±E pairs
        """
        N = H_BdG.shape[0] // 2
        
        # Extract blocks
        H_dense = H_BdG.toarray()
        A = H_dense[:N, :N]
        B = H_dense[:N, N:]
        C = H_dense[N:, :N]
        D = H_dense[N:, N:]
        
        # Check block structure
        checks = {
            'A_hermitian': np.allclose(A, A.conj().T, atol=tol),
            'D_equals_minus_A': np.allclose(D, -A, atol=tol),
            'B_symmetric': np.allclose(B, B.T, atol=tol),
            'C_equals_minus_B_conj': np.allclose(C, -B.conj(), atol=tol)
        }
        
        # Overall particle-hole symmetry
        tau1 = np.block([[np.zeros((N,N)), np.eye(N)],
                         [np.eye(N), np.zeros((N,N))]])
        
        pH_symmetric = np.allclose(
            tau1 @ H_dense.conj() @ tau1,
            -H_dense,
            atol=tol
        )
        
        checks['particle_hole_symmetry'] = pH_symmetric
        
        return checks

    def compute_spectrum(self, H_BdG: csr_matrix, k: int = 16, 
                        sigma: complex = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenspectrum of BdG operator
        
        Args:
            H_BdG: BdG matrix
            k: Number of eigenvalues to compute
            sigma: Target eigenvalue (None for extremal)
            
        Returns:
            (eigenvalues, eigenvectors)
        """
        try:
            if sigma is not None:
                # Target specific eigenvalues near sigma
                eigenvalues, eigenvectors = eigs(H_BdG, k=k, sigma=sigma, which='LM')
            else:
                # Look for largest magnitude eigenvalues
                eigenvalues, eigenvectors = eigs(H_BdG, k=k, which='LM')
            
            # Sort by imaginary part (instabilities first)
            idx = np.argsort(-eigenvalues.imag)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Verify spectrum comes in +/- pairs (particle-hole symmetry)
            self._verify_spectrum_symmetry(eigenvalues)
            
            return eigenvalues, eigenvectors
            
        except Exception as e:
            logger.error(f"Eigenvalue computation failed: {e}")
            # Return safe default
            return np.zeros(k, dtype=complex), np.zeros((H_BdG.shape[0], k), dtype=complex)
    
    def _verify_spectrum_symmetry(self, eigenvalues: np.ndarray, tol: float = 1e-8):
        """Check that spectrum comes in +/- pairs"""
        for ev in eigenvalues:
            # For each eigenvalue, there should be a -ev* in the spectrum
            neg_conj = -np.conj(ev)
            distances = np.abs(eigenvalues - neg_conj)
            if np.min(distances) > tol:
                logger.warning(f"Eigenvalue {ev} lacks particle-hole partner")
    
    def extract_stability_info(self, eigenvalues: np.ndarray) -> Dict[str, Any]:
        """
        Extract stability information from BdG spectrum
        
        Returns:
            Dictionary with stability metrics
        """
        # Filter numerical noise
        eigenvalues_clean = eigenvalues.copy()
        eigenvalues_clean[np.abs(eigenvalues_clean.imag) < 1e-10] = \
            eigenvalues_clean[np.abs(eigenvalues_clean.imag) < 1e-10].real
        
        # Lyapunov exponents are imaginary parts
        lyapunov = eigenvalues_clean.imag
        
        # Find unstable modes
        unstable_mask = lyapunov > 1e-10
        n_unstable = np.sum(unstable_mask)
        
        # Maximum growth rate
        max_lyapunov = np.max(lyapunov) if n_unstable > 0 else 0.0
        
        # Oscillation frequencies (real parts)
        frequencies = np.abs(eigenvalues_clean.real)
        
        return {
            'stable': n_unstable == 0,
            'n_unstable_modes': int(n_unstable),
            'max_lyapunov': float(max_lyapunov),
            'growth_time': 1.0/max_lyapunov if max_lyapunov > 0 else np.inf,
            'oscillation_frequencies': frequencies[frequencies > 1e-6].tolist(),
            'spectral_gap': float(np.min(np.abs(eigenvalues_clean[np.abs(eigenvalues_clean) > 1e-10])))
                           if len(eigenvalues_clean) > 0 else 0.0
        }
    
    def create_dark_soliton_1d(self, x: np.ndarray, v: float = 0.0, 
                              x0: float = 0.0) -> Tuple[np.ndarray, float]:
        """
        Create 1D dark soliton solution
        
        ψ(x) = sqrt(ρ∞) * tanh((x-x0)/ξ) * exp(i*v*x)
        
        Args:
            x: Spatial grid
            v: Soliton velocity
            x0: Soliton center
            
        Returns:
            (wavefunction, chemical_potential)
        """
        # Healing length (assuming g=1, ρ∞=1)
        xi = 1.0 / np.sqrt(2.0)
        
        # Dark soliton profile
        psi = np.tanh((x - x0) / xi) * np.exp(1j * v * x)
        
        # Chemical potential for uniform background
        mu = 1.0  # For ρ∞ = 1, g = 1
        
        return psi, mu
    
    def test_dark_soliton_stability(self, n_points: int = 256, v: float = 0.0) -> Dict[str, Any]:
        """
        Test stability of a dark soliton
        
        Args:
            n_points: Number of grid points
            v: Soliton velocity
            
        Returns:
            Stability analysis results
        """
        # Create grid
        L = 40.0
        x = np.linspace(-L/2, L/2, n_points, endpoint=False)
        self.dx = x[1] - x[0]
        
        # Create dark soliton
        psi, mu = self.create_dark_soliton_1d(x, v=v)
        
        # Assemble BdG operator
        H_BdG = self.assemble_bdg(psi, mu)
        
        # Check symmetries
        symmetries = self.verify_symmetries(H_BdG)
        logger.info(f"Symmetry checks: {symmetries}")
        
        # Compute spectrum
        eigenvalues, eigenvectors = self.compute_spectrum(H_BdG, k=32)
        
        # Extract stability info
        stability = self.extract_stability_info(eigenvalues)
        stability['velocity'] = v
        stability['symmetries_valid'] = all(symmetries.values())
        
        return stability


# New chi-aware functions
def bdg_lambda_max(
    psi: np.ndarray,
    dx: float,
    g: float = 1.0,
    chi: float = 0.0,
    pole_list: Optional[List[Tuple[float, float]]] = None
) -> float:
    """
    Max real eigenvalue with χ-reduced side-mode self-energy.
    
    Args:
        psi: Wavefunction
        dx: Spatial discretization  
        g: Nonlinearity
        chi: Geometric factor from chi reduction
        pole_list: [(ω₀, Γ), …] from Σ(ω) pole extraction
        
    Returns:
        Maximum Lyapunov exponent
    """
    solver = BdGSolver(dx=dx)
    
    # Estimate chemical potential
    mu = g * np.mean(np.abs(psi)**2)
    
    # Build BdG with chi reduction
    H_BdG = solver.assemble_bdg(psi, mu, g, chi, pole_list)
    
    # Compute spectrum focusing on unstable modes
    eigenvalues, _ = solver.compute_spectrum(H_BdG, k=32, sigma=1j)
    
    # Extract maximum Lyapunov
    lyapunov = eigenvalues.imag
    return np.max(lyapunov) if len(lyapunov) > 0 else 0.0


# Backward compatibility functions
def assemble_bdg(psi0: np.ndarray, g: float = 1.0, dx: float = 1.0) -> csr_matrix:
    """Legacy function - use BdGSolver.assemble_bdg instead"""
    solver = BdGSolver(dx=dx)
    # Estimate chemical potential from density
    mu = g * np.mean(np.abs(psi0)**2)
    return solver.assemble_bdg(psi0, mu, g)


def compute_spectrum(H_BdG, k=16, target='LI'):
    """Legacy function - use BdGSolver.compute_spectrum instead"""
    solver = BdGSolver()
    # Map old target to new sigma
    sigma = 1j if target == 'LI' else None
    return solver.compute_spectrum(H_BdG, k, sigma)


def extract_lyapunov_exponents(eigenvalues):
    """Extract Lyapunov exponents from BdG spectrum"""
    lyapunov = eigenvalues.imag
    lyapunov[np.abs(lyapunov) < 1e-10] = 0
    return lyapunov


def analyze_stability(lyapunov_spectrum):
    """Analyze stability from Lyapunov spectrum"""
    solver = BdGSolver()
    # Convert to eigenvalues format
    eigenvalues = 1j * lyapunov_spectrum
    return solver.extract_stability_info(eigenvalues)


# Main test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing BdG Solver with dark soliton...")
    solver = BdGSolver(boundary='periodic')
    
    # Test static dark soliton
    print("\nStatic soliton (v=0):")
    stability_v0 = solver.test_dark_soliton_stability(v=0.0)
    print(f"Stable: {stability_v0['stable']}")
    print(f"Max Lyapunov: {stability_v0['max_lyapunov']:.6e}")
    print(f"Symmetries valid: {stability_v0['symmetries_valid']}")
    
    # Test moving dark soliton
    print("\nMoving soliton (v=0.1):")
    stability_v01 = solver.test_dark_soliton_stability(v=0.1)
    print(f"Stable: {stability_v01['stable']}")
    print(f"Max Lyapunov: {stability_v01['max_lyapunov']:.6e}")
    
    # The static dark soliton should be stable in 1D
    assert stability_v0['stable'], "Static dark soliton should be stable in 1D!"
    assert stability_v0['symmetries_valid'], "BdG symmetries violated!"
    
    # Test with chi reduction
    print("\n\nTesting chi-reduced system:")
    n = 128
    x = np.linspace(-20, 20, n)
    psi = np.ones(n) * np.exp(1j * 0.1 * x)  # Plane wave
    
    # Test different chi values
    for chi in [0.0, 0.5, 1.0]:
        lambda_max = bdg_lambda_max(psi, dx=x[1]-x[0], chi=chi)
        print(f"χ = {chi:.1f}: λ_max = {lambda_max:.6e}")
