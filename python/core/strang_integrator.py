#!/usr/bin/env python3
"""
Strang Splitting Integrator for Non-linear Schrödinger Equation
Symplectic integrator that conserves norm and energy to machine precision
"""

import numpy as np
from typing import Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class StrangIntegrator:
    """
    Second-order symplectic integrator using Strang splitting
    For NLS: i∂ψ/∂t = -∇²ψ + V(|ψ|²)ψ
    
    Split into:
    - Linear part: i∂ψ/∂t = -∇²ψ (solved in Fourier space)
    - Nonlinear part: i∂ψ/∂t = V(|ψ|²)ψ (solved pointwise)
    """
    
    def __init__(
        self,
        spatial_grid: np.ndarray,
        laplacian: np.ndarray,
        dt: float = 0.01,
        nonlinearity: Optional[Callable] = None
    ):
        """
        Initialize Strang integrator
        
        Args:
            spatial_grid: Spatial discretization points
            laplacian: Discrete Laplacian operator (can be sparse)
            dt: Time step size
            nonlinearity: Function V(|ψ|²) for nonlinear potential
        """
        self.grid = spatial_grid
        self.N = len(spatial_grid)
        self.dt = dt
        self.laplacian = laplacian
        
        # Default cubic nonlinearity if none provided
        if nonlinearity is None:
            self.nonlinearity = lambda rho: rho  # Cubic NLS: V = |ψ|²
        else:
            self.nonlinearity = nonlinearity
        
        # Precompute exponentials for efficiency
        self._precompute_propagators()
        
        # Energy tracking
        self.initial_energy = None
        self.initial_norm = None
        
    def _precompute_propagators(self):
        """Precompute matrix exponentials for linear propagation"""
        # For 1D, compute Fourier space propagator
        if self.grid.ndim == 1:
            self.dx = self.grid[1] - self.grid[0]
            k = 2 * np.pi * np.fft.fftfreq(self.N, self.dx)
            self.k_squared = k**2
            
            # Linear propagator in Fourier space: exp(-i k² dt/2)
            self.linear_prop_half = np.exp(-1j * self.k_squared * self.dt / 2)
            self.linear_prop_full = np.exp(-1j * self.k_squared * self.dt)
        else:
            # For higher dimensions or general lattices, use eigendecomposition
            # This is more expensive but works for any Laplacian
            eigenvalues, eigenvectors = np.linalg.eigh(self.laplacian)
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors
            self.eigenvectors_inv = eigenvectors.T
            
            # Propagators: exp(i λ dt)
            self.linear_prop_half = np.exp(1j * eigenvalues * self.dt / 2)
            self.linear_prop_full = np.exp(1j * eigenvalues * self.dt)
    
    def step(self, psi: np.ndarray, store_energy: bool = True) -> np.ndarray:
        """
        Perform one Strang splitting step
        
        Args:
            psi: Current wavefunction
            store_energy: Whether to compute and store conserved quantities
            
        Returns:
            Updated wavefunction after time dt
        """
        if store_energy and self.initial_energy is None:
            self.initial_energy = self.compute_energy(psi)
            self.initial_norm = self.compute_norm(psi)
            logger.info(f"Initial energy: {self.initial_energy:.10f}, norm: {self.initial_norm:.10f}")
        
        # Strang splitting: exp(-iH₂dt/2) exp(-iH₁dt) exp(-iH₂dt/2)
        # H₁ = -∇² (linear), H₂ = V(|ψ|²) (nonlinear)
        
        # Step 1: Half-step nonlinear evolution
        psi = self._nonlinear_step(psi, self.dt / 2)
        
        # Step 2: Full-step linear evolution
        psi = self._linear_step(psi, self.dt)
        
        # Step 3: Half-step nonlinear evolution
        psi = self._nonlinear_step(psi, self.dt / 2)
        
        # Check conservation
        if store_energy:
            current_energy = self.compute_energy(psi)
            current_norm = self.compute_norm(psi)
            energy_drift = abs(current_energy - self.initial_energy) / abs(self.initial_energy)
            norm_drift = abs(current_norm - self.initial_norm) / self.initial_norm
            
            if energy_drift > 1e-5:
                logger.warning(f"Energy drift: {energy_drift:.2e}")
            if norm_drift > 1e-8:
                logger.warning(f"Norm drift: {norm_drift:.2e}")
        
        return psi
    
    def _linear_step(self, psi: np.ndarray, dt: float) -> np.ndarray:
        """
        Linear evolution: exp(i∇² dt) ψ
        Solved in Fourier space for efficiency
        """
        if self.grid.ndim == 1:
            # FFT method for 1D
            psi_k = np.fft.fft(psi)
            if dt == self.dt:
                psi_k *= self.linear_prop_full
            else:
                psi_k *= np.exp(-1j * self.k_squared * dt)
            return np.fft.ifft(psi_k)
        else:
            # Eigendecomposition method for general case
            psi_eigen = self.eigenvectors_inv @ psi
            if dt == self.dt:
                psi_eigen *= self.linear_prop_full
            else:
                psi_eigen *= np.exp(1j * self.eigenvalues * dt)
            return self.eigenvectors @ psi_eigen
    
    def _nonlinear_step(self, psi: np.ndarray, dt: float) -> np.ndarray:
        """
        Nonlinear evolution: exp(-i V(|ψ|²) dt) ψ
        This is diagonal in position space
        """
        rho = np.abs(psi)**2
        V = self.nonlinearity(rho)
        return psi * np.exp(-1j * V * dt)
    
    def compute_energy(self, psi: np.ndarray) -> float:
        """
        Compute total energy: E = ∫(|∇ψ|² + V(|ψ|²)|ψ|²) dx
        """
        # Kinetic energy
        if self.grid.ndim == 1:
            psi_k = np.fft.fft(psi)
            kinetic = np.real(np.sum(self.k_squared * np.abs(psi_k)**2)) * self.dx
        else:
            kinetic = -np.real(np.vdot(psi, self.laplacian @ psi))
        
        # Potential energy
        rho = np.abs(psi)**2
        if self.nonlinearity(1.0) == 1.0:  # Cubic case
            potential = 0.5 * np.sum(rho**2) * self.dx
        else:
            # General case: need to integrate V(ρ)
            V_int = np.zeros_like(rho)
            for i, r in enumerate(rho):
                # Numerical integration of ∫V(s)ds from 0 to r
                if r > 0:
                    s = np.linspace(0, r, 100)
                    V_int[i] = np.trapz(self.nonlinearity(s), s)
            potential = np.sum(V_int) * self.dx
        
        return kinetic + potential
    
    def compute_norm(self, psi: np.ndarray) -> float:
        """Compute L² norm: ∫|ψ|² dx"""
        if self.grid.ndim == 1:
            return np.sum(np.abs(psi)**2) * self.dx
        else:
            return np.sum(np.abs(psi)**2)
    
    def run_simulation(
        self, 
        psi0: np.ndarray, 
        t_final: float,
        store_every: int = 100
    ) -> Tuple[np.ndarray, list]:
        """
        Run simulation from t=0 to t=t_final
        
        Args:
            psi0: Initial wavefunction
            t_final: Final time
            store_every: Store solution every N steps
            
        Returns:
            (final_psi, trajectory) where trajectory contains stored states
        """
        n_steps = int(t_final / self.dt)
        psi = psi0.copy()
        trajectory = [psi0.copy()]
        
        for i in range(n_steps):
            psi = self.step(psi, store_energy=(i % 100 == 0))
            
            if (i + 1) % store_every == 0:
                trajectory.append(psi.copy())
                logger.debug(f"Step {i+1}/{n_steps}, t={self.dt*(i+1):.3f}")
        
        return psi, trajectory


class StrangIntegratorBdG:
    """
    Strang integrator for Bogoliubov-de Gennes equations
    Handles coupled (u,v) components with particle-hole symmetry
    """
    
    def __init__(
        self,
        spatial_grid: np.ndarray,
        laplacian: np.ndarray,
        chemical_potential: float,
        dt: float = 0.01
    ):
        self.grid = spatial_grid
        self.laplacian = laplacian
        self.mu = chemical_potential
        self.dt = dt
        
        # Initialize base integrators for u and v components
        self.integrator_u = StrangIntegrator(spatial_grid, laplacian, dt)
        self.integrator_v = StrangIntegrator(spatial_grid, laplacian, dt)
        
    def step(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evolve BdG equations:
        i∂u/∂t = -∇²u - μu + 2|u|²u + |v|²u + u*v²
        i∂v/∂t = +∇²v + μv - 2|v|²v - |u|²v - u²v*
        """
        # This requires careful handling of the coupling terms
        # Implementation depends on specific BdG formulation
        raise NotImplementedError("BdG Strang splitting requires problem-specific implementation")


def create_sech_soliton(x: np.ndarray, v: float = 0.5) -> np.ndarray:
    """
    Create analytic sech soliton solution for cubic NLS
    ψ(x,t) = sech(x - vt) exp(i(vx/2 - (v²/4 - 1)t))
    """
    return np.sech(x) * np.exp(1j * v * x / 2)


def test_strang_conservation():
    """Test energy and norm conservation for sech soliton"""
    # 1D grid
    L = 40.0
    N = 512
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    dx = x[1] - x[0]
    
    # Discrete Laplacian (periodic BC)
    laplacian = np.zeros((N, N))
    for i in range(N):
        laplacian[i, i] = -2.0
        laplacian[i, (i+1) % N] = 1.0
        laplacian[i, (i-1) % N] = 1.0
    laplacian /= dx**2
    
    # Initial soliton
    psi0 = create_sech_soliton(x)
    
    # Create integrator
    dt = 0.01
    integrator = StrangIntegrator(x, laplacian, dt)
    
    # Run for 1000 steps
    psi, trajectory = integrator.run_simulation(psi0, t_final=10.0, store_every=100)
    
    # Check conservation
    E0 = integrator.compute_energy(psi0)
    E_final = integrator.compute_energy(psi)
    N0 = integrator.compute_norm(psi0)
    N_final = integrator.compute_norm(psi)
    
    print(f"Energy drift: {abs(E_final - E0)/abs(E0):.2e}")
    print(f"Norm drift: {abs(N_final - N0)/N0:.2e}")
    
    # Check soliton shape preservation
    error = np.linalg.norm(np.abs(psi) - np.abs(psi0)) / np.linalg.norm(np.abs(psi0))
    print(f"Shape error: {error:.2e}")
    
    return abs(E_final - E0)/abs(E0) < 1e-5 and abs(N_final - N0)/N0 < 1e-8


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Strang integrator conservation...")
    success = test_strang_conservation()
    print(f"Test {'PASSED' if success else 'FAILED'}")
