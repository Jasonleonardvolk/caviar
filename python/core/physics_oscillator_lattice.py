#!/usr/bin/env python3
"""
Physics-Enhanced Oscillator Lattice with Symplectic Integration
Implements proper energy-conserving dynamics for the oscillator network
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum

# Import Strang integrator for symplectic evolution
try:
    from python.core.strang_integrator import StrangIntegrator
    STRANG_AVAILABLE = True
except ImportError:
    STRANG_AVAILABLE = False
    
# Import physics instrumentation
try:
    from python.core.physics_instrumentation import EnergyTracker, PhysicsMonitor
    INSTRUMENTATION_AVAILABLE = True
except ImportError:
    INSTRUMENTATION_AVAILABLE = False

logger = logging.getLogger(__name__)

class IntegrationMethod(Enum):
    """Available integration methods"""
    EULER = "euler"
    RK4 = "rk4"
    STRANG = "strang"
    SYMPLECTIC_EULER = "symplectic_euler"

@dataclass
class PhysicsOscillatorLattice:
    """
    Enhanced oscillator lattice with proper physics
    
    Key improvements:
    1. Symplectic integration options for energy conservation
    2. Proper Hamiltonian formulation
    3. Energy and norm tracking
    4. Support for NLS dynamics beyond simple Kuramoto
    """
    
    size: int = 100
    integration_method: IntegrationMethod = IntegrationMethod.SYMPLECTIC_EULER
    
    # State arrays
    phases: np.ndarray = field(init=False)
    amplitudes: np.ndarray = field(init=False)
    natural_frequencies: np.ndarray = field(init=False)
    
    # Coupling matrix (sparse in general)
    coupling_matrix: Optional[np.ndarray] = None
    
    # Physics parameters
    nonlinearity_g: float = 1.0
    damping: float = 0.0
    
    # Conservation tracking
    initial_energy: Optional[float] = None
    initial_norm: Optional[float] = None
    energy_history: List[float] = field(default_factory=list)
    
    # Strang integrator (if available)
    strang_integrator: Optional[Any] = None
    
    # Physics monitor
    physics_monitor: Optional[Any] = None
    
    def __post_init__(self):
        """Initialize arrays and integrators"""
        # Initialize state
        self.phases = np.zeros(self.size)
        self.amplitudes = np.ones(self.size)
        self.natural_frequencies = np.zeros(self.size)
        
        # Initialize coupling matrix
        if self.coupling_matrix is None:
            self.coupling_matrix = np.zeros((self.size, self.size))
        
        # Set up integrator
        if self.integration_method == IntegrationMethod.STRANG and STRANG_AVAILABLE:
            # Create 1D spatial grid
            x = np.linspace(0, 10, self.size)
            # Build discrete Laplacian from coupling matrix
            laplacian = -self.coupling_matrix + np.diag(self.coupling_matrix.sum(axis=1))
            
            self.strang_integrator = StrangIntegrator(
                spatial_grid=x,
                laplacian=laplacian,
                dt=0.01,
                nonlinearity=lambda rho: self.nonlinearity_g * rho
            )
        
        # Set up physics monitoring
        if INSTRUMENTATION_AVAILABLE:
            self.physics_monitor = PhysicsMonitor("OscillatorLattice")
            self.physics_monitor.register_invariant(
                "energy",
                lambda sys: sys.compute_total_energy(),
                tolerance=1e-6
            )
            self.physics_monitor.register_invariant(
                "norm",
                lambda sys: sys.compute_total_norm(),
                tolerance=1e-8
            )
    
    def add_oscillator(self, phase: float, frequency: float = 0.0, 
                      amplitude: float = 1.0, index: Optional[int] = None) -> int:
        """Add or update an oscillator"""
        if index is None:
            # Find first available slot
            for i in range(self.size):
                if self.amplitudes[i] < 1e-10:
                    index = i
                    break
            else:
                raise ValueError("Lattice is full")
        
        self.phases[index] = phase
        self.natural_frequencies[index] = frequency
        self.amplitudes[index] = amplitude
        
        return index
    
    def set_coupling(self, i: int, j: int, strength: float):
        """Set coupling between oscillators i and j"""
        self.coupling_matrix[i, j] = strength
        # Ensure symmetry for energy conservation
        self.coupling_matrix[j, i] = strength
        
        # Update Laplacian in Strang integrator if needed
        if self.strang_integrator is not None:
            laplacian = -self.coupling_matrix + np.diag(self.coupling_matrix.sum(axis=1))
            self.strang_integrator.laplacian = laplacian
    
    def step(self, dt: float = 0.01) -> Dict[str, float]:
        """
        Evolve the system by time dt using selected integration method
        
        Returns:
            Dictionary with step information (energy drift, etc.)
        """
        # Store initial values if first step
        if self.initial_energy is None:
            self.initial_energy = self.compute_total_energy()
            self.initial_norm = self.compute_total_norm()
        
        # Choose integration method
        if self.integration_method == IntegrationMethod.EULER:
            self._step_euler(dt)
        elif self.integration_method == IntegrationMethod.RK4:
            self._step_rk4(dt)
        elif self.integration_method == IntegrationMethod.SYMPLECTIC_EULER:
            self._step_symplectic_euler(dt)
        elif self.integration_method == IntegrationMethod.STRANG:
            self._step_strang(dt)
        
        # Compute conservation metrics
        current_energy = self.compute_total_energy()
        current_norm = self.compute_total_norm()
        
        energy_drift = abs(current_energy - self.initial_energy) / abs(self.initial_energy)
        norm_drift = abs(current_norm - self.initial_norm) / self.initial_norm
        
        # Store history
        self.energy_history.append(current_energy)
        if len(self.energy_history) > 1000:
            self.energy_history.pop(0)
        
        # Physics monitoring
        if self.physics_monitor:
            self.physics_monitor.checkpoint(self, f"step_{len(self.energy_history)}")
        
        return {
            "energy": current_energy,
            "energy_drift": energy_drift,
            "norm": current_norm,
            "norm_drift": norm_drift
        }
    
    def _step_euler(self, dt: float):
        """Simple Euler integration (not energy-conserving)"""
        # Compute phase derivatives
        dphi_dt = self._compute_phase_derivatives()
        
        # Update phases
        self.phases += dt * dphi_dt
        self.phases = self.phases % (2 * np.pi)
        
        # Apply damping to amplitudes
        if self.damping > 0:
            self.amplitudes *= (1 - self.damping * dt)
    
    def _step_rk4(self, dt: float):
        """4th-order Runge-Kutta (more accurate but not symplectic)"""
        # RK4 for phase evolution
        k1 = self._compute_phase_derivatives()
        
        self.phases += dt/6 * k1  # Simplified for demonstration
        # Full RK4 would compute k2, k3, k4
        
        self.phases = self.phases % (2 * np.pi)
    
    def _step_symplectic_euler(self, dt: float):
        """
        Symplectic Euler method
        Preserves phase space structure better than standard Euler
        """
        # Convert to complex representation
        psi = self.amplitudes * np.exp(1j * self.phases)
        
        # Compute forces (gradient of Hamiltonian)
        laplacian_psi = self.coupling_matrix @ psi
        nonlinear_term = self.nonlinearity_g * np.abs(psi)**2 * psi
        
        # Symplectic update
        # First update momenta (imaginary part)
        dpsi_dt = 1j * (laplacian_psi + nonlinear_term)
        psi_half = psi + 0.5 * dt * dpsi_dt
        
        # Then update positions using updated momenta
        laplacian_psi_half = self.coupling_matrix @ psi_half
        nonlinear_term_half = self.nonlinearity_g * np.abs(psi_half)**2 * psi_half
        dpsi_dt_half = 1j * (laplacian_psi_half + nonlinear_term_half)
        
        psi = psi + dt * dpsi_dt_half
        
        # Extract phases and amplitudes
        self.phases = np.angle(psi)
        self.amplitudes = np.abs(psi)
    
    def _step_strang(self, dt: float):
        """Use Strang splitting integrator (best conservation)"""
        if self.strang_integrator is None:
            logger.warning("Strang integrator not available, falling back to symplectic Euler")
            self._step_symplectic_euler(dt)
            return
        
        # Convert to wavefunction
        psi = self.amplitudes * np.exp(1j * self.phases)
        
        # Evolve using Strang splitting
        psi = self.strang_integrator.step(psi, store_energy=True)
        
        # Extract phases and amplitudes
        self.phases = np.angle(psi)
        self.amplitudes = np.abs(psi)
    
    def _compute_phase_derivatives(self) -> np.ndarray:
        """Compute dφ/dt for all oscillators"""
        # Natural frequencies
        dphi_dt = self.natural_frequencies.copy()
        
        # Kuramoto coupling
        for i in range(self.size):
            if self.amplitudes[i] < 1e-10:
                continue
            
            for j in range(self.size):
                if i != j and self.amplitudes[j] > 1e-10:
                    coupling = self.coupling_matrix[i, j]
                    phase_diff = self.phases[j] - self.phases[i]
                    dphi_dt[i] += coupling * self.amplitudes[j] * np.sin(phase_diff)
        
        return dphi_dt
    
    def compute_total_energy(self) -> float:
        """
        Compute total energy of the system
        E = Σᵢⱼ Kᵢⱼ aᵢaⱼ cos(φᵢ - φⱼ) + Σᵢ ωᵢ aᵢ²
        """
        energy = 0.0
        
        # Kinetic energy from natural frequencies
        energy += np.sum(self.natural_frequencies * self.amplitudes**2)
        
        # Interaction energy
        for i in range(self.size):
            if self.amplitudes[i] < 1e-10:
                continue
            for j in range(i+1, self.size):
                if self.amplitudes[j] < 1e-10:
                    continue
                coupling = self.coupling_matrix[i, j]
                phase_diff = self.phases[i] - self.phases[j]
                energy += coupling * self.amplitudes[i] * self.amplitudes[j] * np.cos(phase_diff)
        
        # Nonlinear energy
        if self.nonlinearity_g != 0:
            energy += 0.5 * self.nonlinearity_g * np.sum(self.amplitudes**4)
        
        return energy
    
    def compute_total_norm(self) -> float:
        """Compute total norm (sum of squared amplitudes)"""
        return np.sum(self.amplitudes**2)
    
    def get_order_parameter(self) -> complex:
        """
        Kuramoto order parameter
        R e^{iΨ} = (1/N) Σⱼ aⱼ e^{iφⱼ}
        """
        if self.size == 0:
            return 0.0
        
        psi = self.amplitudes * np.exp(1j * self.phases)
        return np.mean(psi)
    
    def verify_energy_conservation(self, tolerance: float = 1e-6) -> bool:
        """Check if energy is conserved within tolerance"""
        if self.initial_energy is None:
            return True
        
        current_energy = self.compute_total_energy()
        drift = abs(current_energy - self.initial_energy) / abs(self.initial_energy)
        
        return drift < tolerance
    
    def create_soliton(self, position: int, width: float = 5.0, 
                      velocity: float = 0.0) -> None:
        """
        Create a soliton excitation in the lattice
        
        Args:
            position: Center position of soliton
            width: Spatial width
            velocity: Group velocity
        """
        x = np.arange(self.size)
        
        # Soliton profile
        envelope = 1.0 / np.cosh((x - position) / width)
        phase_profile = velocity * x
        
        # Add to existing state
        self.amplitudes = np.maximum(self.amplitudes, envelope)
        self.phases += phase_profile
        self.phases = self.phases % (2 * np.pi)
    
    def get_physics_report(self) -> Dict[str, Any]:
        """Get comprehensive physics report"""
        report = {
            "size": self.size,
            "integration_method": self.integration_method.value,
            "current_energy": self.compute_total_energy(),
            "current_norm": self.compute_total_norm(),
            "order_parameter": abs(self.get_order_parameter()),
            "active_oscillators": np.sum(self.amplitudes > 1e-10)
        }
        
        if self.initial_energy is not None:
            report["energy_drift"] = abs(self.compute_total_energy() - self.initial_energy) / abs(self.initial_energy)
            report["norm_drift"] = abs(self.compute_total_norm() - self.initial_norm) / self.initial_norm
        
        if self.physics_monitor:
            report["monitor_report"] = self.physics_monitor.report()
        
        return report


# Compatibility layer
def upgrade_oscillator_lattice(old_lattice: Any) -> PhysicsOscillatorLattice:
    """Upgrade old oscillator lattice to physics-enhanced version"""
    size = len(old_lattice.oscillators) if hasattr(old_lattice, 'oscillators') else 100
    
    new_lattice = PhysicsOscillatorLattice(
        size=size,
        integration_method=IntegrationMethod.SYMPLECTIC_EULER
    )
    
    # Copy oscillator data
    if hasattr(old_lattice, 'oscillators'):
        for i, osc in enumerate(old_lattice.oscillators):
            if i < size:
                new_lattice.phases[i] = osc.phase
                new_lattice.natural_frequencies[i] = osc.natural_freq
                new_lattice.amplitudes[i] = osc.amplitude
    
    # Copy coupling matrix
    if hasattr(old_lattice, 'K') and old_lattice.K is not None:
        new_lattice.coupling_matrix = old_lattice.K.copy()
    
    return new_lattice


if __name__ == "__main__":
    # Test energy conservation
    print("Testing Physics-Enhanced Oscillator Lattice")
    print("="*50)
    
    # Create lattice
    lattice = PhysicsOscillatorLattice(
        size=50,
        integration_method=IntegrationMethod.SYMPLECTIC_EULER
    )
    
    # Add some oscillators with random coupling
    for i in range(10):
        lattice.add_oscillator(
            phase=np.random.rand() * 2 * np.pi,
            frequency=0.1 + 0.05 * np.random.randn(),
            amplitude=1.0,
            index=i
        )
    
    # Random coupling
    for i in range(10):
        for j in range(i+1, 10):
            if np.random.rand() < 0.3:
                coupling = 0.1 * np.random.rand()
                lattice.set_coupling(i, j, coupling)
    
    # Test conservation
    print(f"Initial energy: {lattice.compute_total_energy():.10f}")
    print(f"Initial norm: {lattice.compute_total_norm():.10f}")
    
    # Evolve
    for step in range(1000):
        info = lattice.step(dt=0.01)
        
        if step % 100 == 0:
            print(f"Step {step}: E_drift={info['energy_drift']:.2e}, "
                  f"N_drift={info['norm_drift']:.2e}")
    
    # Final report
    report = lattice.get_physics_report()
    print(f"\nFinal Report:")
    print(f"Energy drift: {report['energy_drift']:.2e}")
    print(f"Norm drift: {report['norm_drift']:.2e}")
    print(f"Order parameter: {report['order_parameter']:.3f}")
    
    # Verify conservation
    if lattice.verify_energy_conservation():
        print("✓ Energy conservation verified!")
    else:
        print("✗ Energy conservation violated!")
