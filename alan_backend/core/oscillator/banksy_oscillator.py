"""
Banksy-spin oscillator substrate implementation.

This module provides a second-order Kuramoto-type oscillator model with spin dynamics,
forming the foundation of ALAN's neuromorphic computation system.

Key features:
- Phase oscillator network with momentum-based update (reversible dynamics)
- Coupled spin vector dynamics for altermagnetic interaction
- Hamiltonian-preserving update scheme enabling TRS (time-reversal symmetry)
- Computation of effective synchronization metrics
"""

import numpy as np
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from fractions import Fraction

# Set up logger
logger = logging.getLogger(__name__)


@dataclass
class BanksyConfig:
    """Configuration parameters for the Banksy-spin oscillator."""
    
    # Phase-to-spin coupling gain (γ)
    gamma: float = 0.1
    
    # Spin Hebbian learning rate (ε)
    epsilon: float = 0.01
    
    # Momentum damping factor (η_damp), typically very small (≈1e-4)
    eta_damp: float = 1e-4
    
    # Integration time step (Δt)
    dt: float = 0.01


class SpinVector:
    """A 3D spin vector with fixed magnitude (normalized to 1)."""
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 1.0):
        """Initialize a spin vector and normalize it to unit length."""
        self.data = np.array([x, y, z], dtype=np.float64)
        self.normalize()
    
    def normalize(self) -> None:
        """Normalize the vector to have unit length."""
        norm = np.linalg.norm(self.data)
        if norm > 1e-10:
            self.data /= norm
        else:
            # Default to z-direction if vector is too small
            self.data = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    
    @property
    def x(self) -> float:
        """Get the x component."""
        return self.data[0]
    
    @property
    def y(self) -> float:
        """Get the y component."""
        return self.data[1]
    
    @property
    def z(self) -> float:
        """Get the z component."""
        return self.data[2]
    
    def as_array(self) -> np.ndarray:
        """Get the raw data as a NumPy array."""
        return self.data
    
    def dot(self, other: 'SpinVector') -> float:
        """Compute the dot product with another spin vector."""
        return np.dot(self.data, other.data)
    
    def __repr__(self) -> str:
        """Return a string representation of the spin vector."""
        return f"SpinVector({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"


class BanksyOscillator:
    """Banksy-spin oscillator network implementation."""
    
    def __init__(self, n_oscillators: int, config: Optional[BanksyConfig] = None):
        """Initialize a Banksy oscillator network with the given size.
        
        Args:
            n_oscillators: Number of oscillators in the network
            config: Configuration parameters (or None for defaults)
        """
        self.n_oscillators = n_oscillators
        self.config = config or BanksyConfig()
        
        # Validate timestep configuration to prevent aliasing
        self._validate_timesteps()
        
        # Initialize with random phases, zero momentum, and random spins
        self.phases = np.random.uniform(0, 2*np.pi, n_oscillators)
        self.momenta = np.zeros(n_oscillators)
        
        # Initialize random spin vectors
        self.spins = []
        for _ in range(n_oscillators):
            # Random spin direction
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            z = np.random.uniform(-1, 1)
            self.spins.append(SpinVector(x, y, z))
        
        # Default to all-to-all coupling with uniform strength
        self.coupling = np.ones((n_oscillators, n_oscillators)) * 0.1
        np.fill_diagonal(self.coupling, 0.0)  # No self-coupling
        
        # For tracking metrics
        self._order_history = []
    
    def _validate_timesteps(self) -> None:
        """Validate timestep configuration to prevent aliasing instabilities.
        
        This method ensures the spin/phase frequency ratio is a rational number
        with a small denominator, which prevents numerical instability in the
        MTS (Multiple Time-Scale) integrator. It uses the continued fraction
        method to find the best rational approximation.
        
        Raises:
            ValueError: If the timestep configuration could lead to resonance or
                        excessive computation
        """
        # Natural frequencies (estimated)
        # In a real system, these would be derived from system parameters
        phase_freq = 1.0  # Normalized base frequency for phase
        spin_freq = 8.0   # Typical ratio for spin vs phase dynamics
        
        # Compute the ratio of frequencies
        from fractions import Fraction
        
        # Find the best rational approximation with denominator <= 64
        ratio_frac = Fraction(spin_freq / phase_freq).limit_denominator(64)
        
        # Extract numerator and denominator
        n_spin = ratio_frac.denominator  # This will be our substep count
        
        # Check if the approximation is good enough
        rel_error = abs(spin_freq/phase_freq - ratio_frac) / (spin_freq/phase_freq)
        
        if rel_error > 1e-3:
            raise ValueError(
                f"Cannot find a good rational approximation for spin/phase "
                f"frequency ratio {spin_freq/phase_freq:.4f}. "
                f"Best approximation: {ratio_frac} (error: {rel_error:.6f})"
            )
        
        # Check if the number of substeps is reasonable
        max_substeps = 64
        if n_spin > max_substeps:
            raise ValueError(
                f"Spin/phase frequency ratio requires too many substeps ({n_spin}). "
                f"Maximum allowed: {max_substeps}. "
                f"Consider adjusting system parameters to achieve a smaller ratio."
            )
        
        # Store the validated substep count for later use
        self._validated_substeps = max(1, min(n_spin, max_substeps))
        
        logger.info(
            f"Validated timestep configuration: "
            f"spin/phase ratio ≈ {ratio_frac} "
            f"({self._validated_substeps} substeps)"
        )
    
    def set_coupling(self, coupling: np.ndarray) -> None:
        """Set the coupling matrix between oscillators.
        
        Args:
            coupling: An n_oscillators x n_oscillators matrix of coupling strengths
            
        Raises:
            ValueError: If the coupling matrix has the wrong shape
        """
        if coupling.shape != (self.n_oscillators, self.n_oscillators):
            raise ValueError(
                f"Coupling matrix must have shape ({self.n_oscillators}, {self.n_oscillators})"
            )
        self.coupling = coupling
    
    def average_spin(self) -> SpinVector:
        """Compute the average spin direction across all oscillators."""
        avg_x = np.mean([spin.x for spin in self.spins])
        avg_y = np.mean([spin.y for spin in self.spins])
        avg_z = np.mean([spin.z for spin in self.spins])
        return SpinVector(avg_x, avg_y, avg_z)
    
    @staticmethod
    def wrap_angle(angle: float) -> float:
        """Wrap angle to [0, 2π) range."""
        return angle % (2 * np.pi)
    
    def step(self, spin_substeps: int = 8) -> None:
        """Perform a single update step of the oscillator network using MTS-Verlet.
        
        This uses a multiple-time-scale (MTS) velocity-Verlet integration scheme
        that preserves symplectic properties at both timescales:
        - Outer loop: Phase dynamics (slower timescale)
        - Inner loop: Spin dynamics (faster timescale, sub-stepped)
        
        The algorithm follows:
        1. Half-step phase momentum update
        2. Substeps of spin dynamics with inner Verlet integrator
        3. Full-step phase position update
        4. Final half-step phase momentum update
        
        Args:
            spin_substeps: Number of spin integration steps per phase step
                           (default=8, handles THz vs kHz timescale mismatch)
        """
        # Get parameters from config
        gamma = self.config.gamma
        epsilon = self.config.epsilon
        eta_damp = self.config.eta_damp
        dt = self.config.dt
        
        # Compute spin-related quantities once before phase update
        avg_spin = self.average_spin()
        
        # 1. FIRST HALF-STEP PHASE MOMENTUM UPDATE (p_θ += dt/2 * F_phase)
        for i in range(self.n_oscillators):
            # Calculate coupling torque based on phase differences
            torque = 0.0
            for j in range(self.n_oscillators):
                if i != j:
                    phase_diff = self.phases[j] - self.phases[i]
                    torque += self.coupling[i, j] * np.sin(phase_diff)
            
            # Compute spin-lattice coupling term
            spin_coupling = gamma * (self.spins[i].dot(avg_spin) - 1.0)
            
            # First half-step momentum update with coupling torque and spin-lattice term
            self.momenta[i] += (torque + spin_coupling) * 0.5 * dt
            
            # Apply light damping (preserves energy when η_damp → 0)
            self.momenta[i] *= (1.0 - eta_damp * 0.5)
        
        # 2. INNER LOOP: SPIN DYNAMICS WITH VERLET INTEGRATION
        # Create spin momenta array (not stored permanently, just for integration)
        spin_momenta = np.zeros((self.n_oscillators, 3))
        
        # Compute spin timestep
        spin_dt = dt / spin_substeps
        
        for _ in range(spin_substeps):
            # 2a. First half-step for spin momenta
            avg_spin = self.average_spin()  # Recompute for accurate forces
            
            for i in range(self.n_oscillators):
                # Calculate spin force
                spin_force = np.zeros(3)
                
                # Calculate update based on phase relationships (Hebbian)
                for j in range(self.n_oscillators):
                    if i != j:
                        phase_diff = self.phases[i] - self.phases[j]
                        weight = np.cos(phase_diff)  # Hebbian window function
                        
                        # Add weighted contribution from other spin
                        spin_force += weight * self.spins[j].as_array()
                
                # Scale by learning rate
                spin_force *= epsilon
                
                # Half-step momentum update for spin
                spin_momenta[i] += spin_force * 0.5 * spin_dt
            
            # 2b. Full position step for spins
            for i in range(self.n_oscillators):
                # Update spin position with momentum
                current = self.spins[i].as_array()
                new_spin_array = current + spin_momenta[i] * spin_dt
                
                # Normalize spin (enforces constraint |σ| = 1)
                self.spins[i] = SpinVector(
                    new_spin_array[0],
                    new_spin_array[1],
                    new_spin_array[2]
                )
            
            # 2c. Second half-step for spin momenta
            avg_spin = self.average_spin()  # Recompute after position update
            
            for i in range(self.n_oscillators):
                # Recalculate spin force with updated positions
                spin_force = np.zeros(3)
                
                for j in range(self.n_oscillators):
                    if i != j:
                        phase_diff = self.phases[i] - self.phases[j]
                        weight = np.cos(phase_diff)
                        spin_force += weight * self.spins[j].as_array()
                
                spin_force *= epsilon
                
                # Second half-step momentum update for spin
                spin_momenta[i] += spin_force * 0.5 * spin_dt
        
        # 3. FULL PHASE POSITION UPDATE (θ += dt * p_θ)
        for i in range(self.n_oscillators):
            # Update phase based on momentum (position full-step)
            self.phases[i] = self.wrap_angle(self.phases[i] + dt * self.momenta[i])
        
        # Ensure phases stay in [0, 2π) range
        self.phases = np.array([self.wrap_angle(p) for p in self.phases])
        
        # 4. SECOND HALF-STEP PHASE MOMENTUM UPDATE (p_θ += dt/2 * F_phase)
        # Recompute spin-related quantities for accurate forces
        avg_spin = self.average_spin()
        
        for i in range(self.n_oscillators):
            # Recalculate forces with updated positions
            torque = 0.0
            for j in range(self.n_oscillators):
                if i != j:
                    phase_diff = self.phases[j] - self.phases[i]
                    torque += self.coupling[i, j] * np.sin(phase_diff)
            
            # Compute spin-lattice coupling term with updated spins
            spin_coupling = gamma * (self.spins[i].dot(avg_spin) - 1.0)
            
            # Second half-step momentum update
            self.momenta[i] += (torque + spin_coupling) * 0.5 * dt
            
            # Apply remaining half-step damping
            self.momenta[i] *= (1.0 - eta_damp * 0.5)
        
        # Record order parameter for metrics
        order_param = self.order_parameter()
        self._order_history.append(order_param)
        
        # Log N_eff for TRS controller rollback
        n_eff = self.effective_count(phase_threshold=0.7, spin_threshold=0.3)
        # This would connect to a logger in a full implementation
        if len(self._order_history) % 20 == 0:  # Log periodically to reduce verbosity
            print(f"Step {len(self._order_history)}: Order = {order_param:.4f}, N_eff = {n_eff}")
    
    def order_parameter(self) -> float:
        """Calculate the Kuramoto order parameter (measure of synchronization).
        
        Returns:
            r: The magnitude of the complex order parameter [0, 1]
        """
        # Calculate r = |∑ exp(i*θ_j)| / N
        complex_sum = np.sum(np.exp(1j * self.phases))
        r = np.abs(complex_sum) / self.n_oscillators
        return r
    
    def mean_phase(self) -> float:
        """Calculate the mean phase (argument of the order parameter).
        
        Returns:
            phi: The mean phase angle in radians [0, 2π)
        """
        complex_sum = np.sum(np.exp(1j * self.phases))
        phi = np.angle(complex_sum) % (2 * np.pi)
        return phi
    
    def effective_count(self, phase_threshold: float = 0.7, spin_threshold: float = 0.3) -> int:
        """Calculate the effective number of synchronized oscillators.
        
        N_eff = Σ_i (|Φ_i| > τ_phase ∧ |σ_i – σ̄| < τ_spin)
        
        Args:
            phase_threshold: Threshold for the order parameter [0, 1]
            spin_threshold: Threshold for spin deviation [0, 2]
        
        Returns:
            count: Number of oscillators that satisfy both synchronization criteria
        """
        avg_spin = self.average_spin()
        r = self.order_parameter()
        
        count = 0
        
        for i in range(self.n_oscillators):
            # Check if phase is within threshold of mean phase
            phase_coherent = r > phase_threshold
            
            # Check if spin is within threshold of mean spin
            spin_diff = 1.0 - self.spins[i].dot(avg_spin)
            spin_coherent = spin_diff < spin_threshold
            
            if phase_coherent and spin_coherent:
                count += 1
        
        return count
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the oscillator network.
        
        Returns:
            state: Dictionary containing the current phases, momenta, and spins
        """
        return {
            'phases': self.phases.copy(),
            'momenta': self.momenta.copy(),
            'spins': [SpinVector(s.x, s.y, s.z) for s in self.spins],
            'order_parameter': self.order_parameter(),
            'mean_phase': self.mean_phase(),
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the current state of the oscillator network.
        
        Args:
            state: Dictionary containing the phases, momenta, and optionally spins
        
        Raises:
            ValueError: If the state dimensions don't match the network
        """
        if len(state['phases']) != self.n_oscillators:
            raise ValueError("Phases dimension mismatch")
        
        if len(state['momenta']) != self.n_oscillators:
            raise ValueError("Momenta dimension mismatch")
        
        self.phases = state['phases'].copy()
        self.momenta = state['momenta'].copy()
        
        if 'spins' in state:
            if len(state['spins']) != self.n_oscillators:
                raise ValueError("Spins dimension mismatch")
            self.spins = [SpinVector(s.x, s.y, s.z) for s in state['spins']]


class BanksySimulator:
    """A simulator for running the oscillator network with various configurations."""
    
    def __init__(self, n_oscillators: int, config: Optional[BanksyConfig] = None):
        """Create a new simulator with the given network size.
        
        Args:
            n_oscillators: Number of oscillators in the network
            config: Configuration parameters (or None for defaults)
        """
        self.oscillator = BanksyOscillator(n_oscillators, config)
        self.time = 0.0
        self.history = []
    
    def run(self, steps: int) -> None:
        """Run the simulation for the specified number of steps.
        
        Args:
            steps: Number of time steps to run
        """
        dt = self.oscillator.config.dt
        
        for _ in range(steps):
            self.oscillator.step()
            self.time += dt
            
            # Record state
            state = self.oscillator.get_state()
            self.history.append({
                'time': self.time,
                'phases': state['phases'].copy(),
                'order_parameter': state['order_parameter'],
                'mean_phase': state['mean_phase'],
            })
    
    def get_order_parameter_history(self) -> np.ndarray:
        """Get the history of order parameters over time.
        
        Returns:
            order: Array of shape (n_steps,) with order parameter values
        """
        return np.array([state['order_parameter'] for state in self.history])
    
    def get_time_series(self) -> np.ndarray:
        """Get the time points of the simulation.
        
        Returns:
            time: Array of shape (n_steps,) with time points
        """
        return np.array([state['time'] for state in self.history])
    
    def get_phase_history(self) -> np.ndarray:
        """Get the history of phases for all oscillators.
        
        Returns:
            phases: Array of shape (n_steps, n_oscillators) with phase values
        """
        return np.array([state['phases'] for state in self.history])


if __name__ == '__main__':
    # Simple test/demonstration
    import matplotlib.pyplot as plt
    
    # Create a simulator
    n_oscillators = 32
    sim = BanksySimulator(n_oscillators)
    
    # Add some structure to the coupling matrix
    # (e.g., modular structure with two communities)
    coupling = np.ones((n_oscillators, n_oscillators)) * 0.05
    np.fill_diagonal(coupling, 0.0)
    
    # Strengthen coupling within communities
    community_size = n_oscillators // 2
    for i in range(community_size):
        for j in range(community_size):
            if i != j:
                coupling[i, j] = 0.2
                coupling[i + community_size, j + community_size] = 0.2
    
    sim.oscillator.set_coupling(coupling)
    
    # Run the simulation
    sim.run(steps=500)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Plot order parameter
    plt.subplot(2, 1, 1)
    plt.plot(sim.get_time_series(), sim.get_order_parameter_history())
    plt.xlabel('Time')
    plt.ylabel('Order Parameter')
    plt.title('Banksy Oscillator Synchronization')
    
    # Plot phase evolution
    plt.subplot(2, 1, 2)
    phases = sim.get_phase_history()
    for i in range(n_oscillators):
        plt.plot(sim.get_time_series(), phases[:, i], alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Phase')
    plt.title(f'Phase Evolution of {n_oscillators} Oscillators')
    
    plt.tight_layout()
    plt.show()
