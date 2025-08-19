"""
Spectral Koopman-Kuramoto Oscillator Network - Optimized Version
Maps audio features to collective phase dynamics for holographic visualization
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Protocol
from dataclasses import dataclass, field
import logging
from functools import lru_cache
import threading
from collections import deque
from numba import njit, prange
import warnings

logger = logging.getLogger(__name__)

# Type definitions
PhaseArray = np.ndarray  # Shape (n,) of phases in [0, 2π)
FrequencyArray = np.ndarray  # Shape (n,) of frequencies in rad/s
SpatialFrequencyArray = np.ndarray  # Shape (n, 2) of spatial frequencies


@dataclass
class OscillatorState:
    """Complete state of the oscillator network"""
    phases: PhaseArray          # Individual oscillator phases [0, 2π)
    natural_freqs: FrequencyArray   # Natural frequencies ωᵢ (rad/s)
    coupling: float             # Coupling strength K
    coherence: float            # Order parameter magnitude r ∈ [0,1]
    collective_phase: float     # Collective phase ψ ∈ [0,2π)
    amplitudes: Optional[np.ndarray] = None  # Pre-computed amplitudes


class OscillatorDynamics(Protocol):
    """Protocol for oscillator dynamics implementations"""
    def step(self, state: OscillatorState, dt: float) -> OscillatorState:
        """Advance oscillator state by dt"""
        ...
    
    def compute_order_parameter(self, phases: PhaseArray) -> Tuple[float, float]:
        """Compute Kuramoto order parameter (r, ψ)"""
        ...


# Numba-accelerated functions for performance
@njit(parallel=True, cache=True)
def kuramoto_step_sparse(phases: np.ndarray, natural_freqs: np.ndarray, 
                        coupling: float, dt: float, noise_strength: float,
                        neighbors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Optimized Kuramoto dynamics with sparse coupling.
    
    Args:
        phases: Current phases
        natural_freqs: Natural frequencies
        coupling: Coupling strength
        dt: Time step
        noise_strength: Noise amplitude
        neighbors: (n_edges, 2) array of coupled oscillator pairs
        weights: (n_edges,) array of coupling weights
    
    Returns:
        Updated phases
    """
    n = len(phases)
    new_phases = phases.copy()
    coupling_term = np.zeros(n)
    
    # Compute sparse coupling (much faster than O(N²))
    for edge_idx in prange(len(neighbors)):
        i, j = neighbors[edge_idx]
        weight = weights[edge_idx]
        
        # sin(θⱼ - θᵢ) coupling
        phase_diff = np.sin(phases[j] - phases[i])
        coupling_term[i] += weight * phase_diff
        coupling_term[j] -= weight * phase_diff  # Newton's third law
    
    # Update phases
    for i in prange(n):
        noise = noise_strength * (np.random.randn() * 2 - 1)
        dtheta = natural_freqs[i] + coupling * coupling_term[i] + noise
        new_phases[i] = (phases[i] + dtheta * dt) % (2 * np.pi)
    
    return new_phases


@njit(cache=True)
def compute_order_parameter_fast(phases: np.ndarray) -> Tuple[float, float]:
    """Fast computation of Kuramoto order parameter"""
    n = len(phases)
    real_sum = 0.0
    imag_sum = 0.0
    
    for i in range(n):
        real_sum += np.cos(phases[i])
        imag_sum += np.sin(phases[i])
    
    real_sum /= n
    imag_sum /= n
    
    r = np.sqrt(real_sum**2 + imag_sum**2)
    psi = np.arctan2(imag_sum, real_sum) % (2 * np.pi)
    
    return r, psi


@njit(cache=True)
def precompute_amplitudes(natural_freqs: np.ndarray, coherence: float) -> np.ndarray:
    """Pre-compute oscillator amplitudes for GPU shader"""
    n = len(natural_freqs)
    amplitudes = np.zeros(n)
    
    for i in range(n):
        freq_mag = np.abs(natural_freqs[i]) / (2 * np.pi * 1000)  # Normalize
        amplitudes[i] = np.exp(-freq_mag * 0.1) * coherence
    
    return amplitudes


class SparseCouplingTopology:
    """Manages sparse coupling topology for efficient computation"""
    
    def __init__(self, n_osc: int, topology: str = "nearest_neighbor"):
        self.n = n_osc
        self.topology = topology
        self._build_topology()
    
    def _build_topology(self) -> None:
        """Build coupling topology"""
        if self.topology == "nearest_neighbor":
            # Ring topology with nearest neighbors
            edges = []
            weights = []
            for i in range(self.n):
                # Connect to neighbors
                edges.append([i, (i + 1) % self.n])
                edges.append([i, (i - 1) % self.n])
                weights.extend([1.0, 1.0])
            
            self.edges = np.array(edges, dtype=np.int32)
            self.weights = np.array(weights, dtype=np.float32) / 2.0  # Normalize
            
        elif self.topology == "small_world":
            # Watts-Strogatz small world
            self._build_small_world()
            
        elif self.topology == "all_to_all":
            # Full coupling (warning: O(N²))
            edges = []
            weights = []
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    edges.append([i, j])
                    weights.append(1.0 / self.n)
            
            self.edges = np.array(edges, dtype=np.int32)
            self.weights = np.array(weights, dtype=np.float32)
        
        else:
            raise ValueError(f"Unknown topology: {self.topology}")
    
    def _build_small_world(self, k: int = 4, p: float = 0.1) -> None:
        """Build Watts-Strogatz small world topology"""
        # Start with ring lattice
        edges = []
        for i in range(self.n):
            for j in range(1, k // 2 + 1):
                edges.append([i, (i + j) % self.n])
                edges.append([i, (i - j) % self.n])
        
        # Rewire with probability p
        edges_array = np.array(edges)
        n_edges = len(edges_array)
        
        for idx in range(n_edges):
            if np.random.rand() < p:
                i, j = edges_array[idx]
                # Rewire j to random node
                new_j = np.random.randint(0, self.n)
                if new_j != i:
                    edges_array[idx, 1] = new_j
        
        # Remove duplicates
        unique_edges = set()
        for edge in edges_array:
            unique_edges.add(tuple(sorted(edge)))
        
        self.edges = np.array(list(unique_edges), dtype=np.int32)
        self.weights = np.ones(len(self.edges), dtype=np.float32) / k


class KuramotoDynamics:
    """Optimized Kuramoto dynamics implementation"""
    
    def __init__(self, coupling_topology: SparseCouplingTopology):
        self.topology = coupling_topology
    
    def step(self, state: OscillatorState, dt: float, noise_strength: float = 0.01) -> OscillatorState:
        """Advance oscillator state using sparse coupling"""
        new_phases = kuramoto_step_sparse(
            state.phases,
            state.natural_freqs,
            state.coupling,
            dt,
            noise_strength,
            self.topology.edges,
            self.topology.weights
        )
        
        # Compute new order parameter
        r, psi = compute_order_parameter_fast(new_phases)
        
        # Pre-compute amplitudes if needed
        amplitudes = precompute_amplitudes(state.natural_freqs, r)
        
        return OscillatorState(
            phases=new_phases,
            natural_freqs=state.natural_freqs.copy(),
            coupling=state.coupling,
            coherence=r,
            collective_phase=psi,
            amplitudes=amplitudes
        )
    
    def compute_order_parameter(self, phases: PhaseArray) -> Tuple[float, float]:
        """Compute Kuramoto order parameter"""
        return compute_order_parameter_fast(phases)


class HolographicProjector:
    """Handles holographic wavefield computations"""
    
    def __init__(self, n_osc: int):
        self.n = n_osc
        self._spatial_freq_cache: Dict[str, SpatialFrequencyArray] = {}
    
    @lru_cache(maxsize=128)
    def compute_spatial_frequencies(self, 
                                  natural_freqs_tuple: Tuple[float, ...],
                                  arrangement: str = "spiral") -> SpatialFrequencyArray:
        """Compute spatial frequencies with caching"""
        key = f"{arrangement}_{hash(natural_freqs_tuple)}"
        
        if key in self._spatial_freq_cache:
            return self._spatial_freq_cache[key]
        
        natural_freqs = np.array(natural_freqs_tuple)
        spatial_freqs = np.zeros((self.n, 2))
        
        if arrangement == "spiral":
            angles = np.arange(self.n) * np.pi / self.n
            freq_normalized = natural_freqs / (2 * np.pi * 1000)
            spiral_factor = 1 + 0.1 * np.arange(self.n)
            
            spatial_freqs[:, 0] = freq_normalized * np.cos(angles) * spiral_factor
            spatial_freqs[:, 1] = freq_normalized * np.sin(angles) * spiral_factor
            
        elif arrangement == "grid":
            grid_size = int(np.ceil(np.sqrt(self.n)))
            for i in range(self.n):
                x = (i % grid_size) / grid_size - 0.5
                y = (i // grid_size) / grid_size - 0.5
                freq_mag = natural_freqs[i] / (2 * np.pi * 1000)
                spatial_freqs[i] = [x * freq_mag, y * freq_mag]
        
        self._spatial_freq_cache[key] = spatial_freqs
        return spatial_freqs
    
    def get_wavefield_params(self, state: OscillatorState) -> Dict[str, Any]:
        """Get parameters for holographic wavefield modulation"""
        # Convert to tuple for caching
        freq_tuple = tuple(state.natural_freqs.tolist())
        spatial_freqs = self.compute_spatial_frequencies(freq_tuple)
        
        # Extend to 32 oscillators for shader compatibility
        extended_phases = np.zeros(32)
        extended_phases[:self.n] = state.phases
        
        extended_spatial_freqs = np.zeros((32, 2))
        extended_spatial_freqs[:self.n] = spatial_freqs
        
        extended_amplitudes = np.zeros(32)
        if state.amplitudes is not None:
            extended_amplitudes[:self.n] = state.amplitudes
        else:
            extended_amplitudes[:self.n] = precompute_amplitudes(
                state.natural_freqs, state.coherence
            )
        
        return {
            'phase_modulation': float(state.collective_phase),
            'coherence': float(state.coherence),
            'oscillator_phases': extended_phases.tolist(),
            'dominant_freq': float(np.mean(state.natural_freqs) / (2 * np.pi)),
            'spatial_frequencies': extended_spatial_freqs.tolist(),
            'amplitudes': extended_amplitudes.tolist()
        }


class BanksyOscillator:
    """
    Refactored Kuramoto oscillator with separated concerns.
    Handles only orchestration, delegating dynamics and holography.
    """
    
    def __init__(self, 
                 n_osc: int = 12,
                 bandwidth: float = 100.0,
                 dt: float = 0.1,
                 noise_strength: float = 0.01,
                 topology: str = "nearest_neighbor"):
        """
        Initialize oscillator network.
        
        Args:
            n_osc: Number of oscillators
            bandwidth: Frequency spread around centroid (Hz)
            dt: Integration timestep (seconds)
            noise_strength: Phase noise for realistic dynamics
            topology: Coupling topology ("nearest_neighbor", "small_world", "all_to_all")
        """
        self.n = n_osc
        self.bandwidth = bandwidth
        self.dt = dt
        self.noise_strength = noise_strength
        
        # Initialize subsystems
        self.coupling_topology = SparseCouplingTopology(n_osc, topology)
        self.dynamics = KuramotoDynamics(self.coupling_topology)
        self.projector = HolographicProjector(n_osc)
        
        # Initialize state
        self.state = OscillatorState(
            phases=np.random.uniform(0, 2*np.pi, self.n),
            natural_freqs=np.zeros(self.n),
            coupling=0.1,
            coherence=0.0,
            collective_phase=0.0
        )
        
        # Coupling parameters
        self.K_min = 0.1
        self.K_max = 1.0
        
        # History for smoothing
        self.coherence_history = deque(maxlen=10)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Adaptation parameters
        self.freq_adaptation_rate = 0.1
        self.target_freqs = np.zeros(self.n)
        
        logger.info(f"Initialized BanksyOscillator: n={n_osc}, topology={topology}")
    
    def map_parameters(self, 
                      centroid: float, 
                      emotion_intensity: float,
                      rms: Optional[float] = None) -> None:
        """Map audio features to oscillator parameters (thread-safe)"""
        with self._lock:
            self._map_parameters_unsafe(centroid, emotion_intensity, rms)
    
    def _map_parameters_unsafe(self, 
                              centroid: float, 
                              emotion_intensity: float,
                              rms: Optional[float] = None) -> None:
        """Internal parameter mapping without locking"""
        # Create frequency distribution
        if self.n == 1:
            deltas = np.array([0.0])
        else:
            deltas = np.linspace(-self.bandwidth, self.bandwidth, self.n)
        
        # Convert to rad/s
        self.target_freqs = 2 * np.pi * (centroid + deltas)
        
        # Smooth frequency adaptation
        self.state.natural_freqs = (
            (1 - self.freq_adaptation_rate) * self.state.natural_freqs +
            self.freq_adaptation_rate * self.target_freqs
        )
        
        # Update coupling strength
        K = self.K_min + emotion_intensity * (self.K_max - self.K_min)
        
        # Optional RMS modulation
        if rms is not None:
            amplitude_boost = np.clip(rms * 2, 0, 1)
            K *= (1 + 0.5 * amplitude_boost)
        
        self.state = OscillatorState(
            phases=self.state.phases,
            natural_freqs=self.state.natural_freqs,
            coupling=K,
            coherence=self.state.coherence,
            collective_phase=self.state.collective_phase,
            amplitudes=self.state.amplitudes
        )
        
        logger.debug(f"Parameters mapped: f0={centroid:.1f}Hz, K={K:.3f}")
    
    def step(self, dt: Optional[float] = None) -> None:
        """Advance oscillator dynamics (thread-safe)"""
        if dt is None:
            dt = self.dt
        
        with self._lock:
            # Delegate to dynamics engine
            self.state = self.dynamics.step(self.state, dt, self.noise_strength)
            
            # Update history
            self.coherence_history.append(self.state.coherence)
    
    def compute_order_parameter(self) -> Tuple[float, float]:
        """Get current order parameter"""
        with self._lock:
            # Use smoothed coherence
            r_smooth = np.mean(list(self.coherence_history)) if self.coherence_history else self.state.coherence
            return r_smooth, self.state.collective_phase
    
    def psi_state(self) -> Dict[str, Any]:
        """Get oscillator state for downstream processing"""
        with self._lock:
            r, psi = self.compute_order_parameter()
            
            state = {
                "phase_coherence": float(r),
                "psi_phase": float(psi),
                "psi_magnitude": float(r),
                "oscillator_phases": self.state.phases.tolist(),
                "oscillator_frequencies": (self.state.natural_freqs / (2 * np.pi)).tolist(),
                "coupling_strength": float(self.state.coupling),
                "dominant_frequency": float(np.mean(self.state.natural_freqs) / (2 * np.pi))
            }
            
            return state
    
    def get_wavefield_params(self) -> Dict[str, Any]:
        """Get parameters for holographic wavefield modulation"""
        with self._lock:
            return self.projector.get_wavefield_params(self.state)
    
    def reset(self) -> None:
        """Reset oscillator to initial conditions"""
        with self._lock:
            self.state = OscillatorState(
                phases=np.random.uniform(0, 2*np.pi, self.n),
                natural_freqs=np.zeros(self.n),
                coupling=self.K_min,
                coherence=0.0,
                collective_phase=0.0
            )
            self.coherence_history.clear()
            logger.info("Oscillator reset")
    
    def set_phases(self, phases: List[float]) -> None:
        """Manually set oscillator phases"""
        if len(phases) != self.n:
            raise ValueError(f"Expected {self.n} phases, got {len(phases)}")
        
        with self._lock:
            self.state.phases = np.array(phases) % (2 * np.pi)


# Specialized variants with optimized parameters
class VoiceOscillator(BanksyOscillator):
    """Oscillator tuned for voice/speech processing"""
    
    def __init__(self, n_osc: int = 8):
        super().__init__(
            n_osc=n_osc,
            bandwidth=50.0,
            dt=0.05,
            noise_strength=0.02,
            topology="small_world"  # Better for speech patterns
        )
        self.K_min = 0.2
        self.K_max = 0.8


class MusicOscillator(BanksyOscillator):
    """Oscillator tuned for music processing"""
    
    def __init__(self, n_osc: int = 16):
        super().__init__(
            n_osc=n_osc,
            bandwidth=200.0,
            dt=0.02,
            noise_strength=0.005,
            topology="nearest_neighbor"  # Good for harmonic structures
        )
        self.K_min = 0.3
        self.K_max = 1.5


# Factory function
def create_oscillator_for_context(context: str = "general", 
                                n_osc: Optional[int] = None) -> BanksyOscillator:
    """Factory function to create appropriate oscillator type"""
    if context == "voice":
        return VoiceOscillator(n_osc or 8)
    elif context == "music":
        return MusicOscillator(n_osc or 16)
    else:
        return BanksyOscillator(n_osc or 12)


# Performance testing utilities
def benchmark_oscillator(n_osc: int = 16, n_steps: int = 1000) -> Dict[str, float]:
    """Benchmark oscillator performance"""
    import time
    
    results = {}
    
    for topology in ["nearest_neighbor", "small_world"]:
        osc = BanksyOscillator(n_osc=n_osc, topology=topology)
        osc.map_parameters(440.0, 0.7, 0.5)
        
        start = time.time()
        for _ in range(n_steps):
            osc.step()
        end = time.time()
        
        results[topology] = (end - start) / n_steps * 1000  # ms per step
    
    return results


if __name__ == "__main__":
    # Test optimized implementation
    print("Testing optimized oscillator...")
    
    # Benchmark different configurations
    for n in [8, 16, 32, 64]:
        print(f"\nBenchmarking with {n} oscillators:")
        results = benchmark_oscillator(n_osc=n, n_steps=100)
        for topology, time_ms in results.items():
            print(f"  {topology}: {time_ms:.3f} ms/step")
    
    # Test basic functionality
    osc = create_oscillator_for_context("music")
    osc.map_parameters(440.0, 0.8, 0.7)
    
    for _ in range(10):
        osc.step()
    
    state = osc.psi_state()
    print(f"\nFinal state: coherence={state['phase_coherence']:.3f}")
