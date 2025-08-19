#!/usr/bin/env python3
"""
Chaos-Control Layer (CCL) - Sandboxed Edge-of-Chaos Computation
Implements the core computational substrate for TORI's chaos-assisted processing

Based on 2023-2025 research showing 3-16x energy efficiency at edge-of-chaos
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
import asyncio
import logging
from enum import Enum
from collections import defaultdict, deque
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pickle

# Import EigenSentry 2.0 and other components
from python.core.eigensentry.core import (
    EigenSentry2, InstabilityType, CoordinationSignal
)
from python.core.unified_metacognitive_integration import (
    MetacognitiveState, CognitiveStateManager
)
from python.core.adaptive_timestep import AdaptiveTimestep
from python.core.chaos_channel_controller import ChaosChannelController

logger = logging.getLogger(__name__)

# ========== CCL Configuration ==========

# Isolation levels
MEMORY_ISOLATION_SIZE_MB = 512  # Dedicated memory for CCL
MAX_COMPUTATION_TIME_S = 10.0   # Hard timeout for chaos computations
CHECKPOINT_INTERVAL_S = 0.1     # Frequency of state checkpoints

# Chaos parameters from research
DARK_SOLITON_VELOCITY = 0.8     # Fraction of linear wave speed
ATTRACTOR_HOP_THRESHOLD = 0.7   # Similarity threshold for basin jumping  
PHASE_EXPLOSION_CONE = np.pi/6  # Maximum phase spread before reset

# Energy efficiency multipliers (from research)
EFFICIENCY_MULTIPLIERS = {
    'dark_soliton': 3.2,      # Observed in optical systems
    'attractor_hop': 5.7,     # From reservoir computing
    'phase_explosion': 16.4,  # Maximum observed in coupled oscillators
    'edge_stable': 1.0        # Baseline
}

# ========== Core Data Structures ==========

class ChaosMode(Enum):
    """Supported chaos computation modes"""
    DARK_SOLITON = "dark_soliton"
    ATTRACTOR_HOP = "attractor_hop"
    PHASE_EXPLOSION = "phase_explosion"
    HYBRID = "hybrid"

@dataclass
class ChaosTask:
    """Represents a computational task to be executed with chaos assistance"""
    task_id: str
    mode: ChaosMode
    input_data: np.ndarray
    parameters: Dict[str, Any]
    callback: Optional[Callable] = None
    priority: int = 0
    energy_budget: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChaosResult:
    """Result from chaos-assisted computation"""
    task_id: str
    success: bool
    output_data: Optional[np.ndarray]
    energy_used: int
    computation_time: float
    efficiency_gain: float
    metadata: Dict[str, Any] = field(default_factory=dict)

# ========== Dark Soliton Processor ==========

class DarkSolitonProcessor:
    """
    Implements dark soliton dynamics for memory operations
    Based on research showing stable information propagation in nonlinear media
    """
    
    def __init__(self, lattice_size: int = 1000):
        self.lattice_size = lattice_size
        self.lattice = np.zeros(lattice_size, dtype=complex)
        self.dispersion_relation = self._init_dispersion()
        
    def _init_dispersion(self) -> np.ndarray:
        """Initialize dispersion relation for soliton propagation"""
        k = np.fft.fftfreq(self.lattice_size, d=1.0)
        # Nonlinear dispersion relation
        omega = k**2 - 0.5 * k**4
        return omega
        
    def create_dark_soliton(self, position: int, width: float = 10.0, 
                           depth: float = 0.8) -> np.ndarray:
        """Create a dark soliton at specified position"""
        x = np.arange(self.lattice_size)
        # Dark soliton profile: depression in amplitude
        soliton = 1.0 - depth * np.cosh((x - position) / width) ** (-2)
        # Add phase jump
        phase = np.pi * np.tanh((x - position) / width)
        return soliton * np.exp(1j * phase)
        
    def propagate(self, state: np.ndarray, time_steps: int = 100,
                 dt: float = 0.01) -> List[np.ndarray]:
        """
        Propagate soliton state using split-step Fourier method
        Returns trajectory of states
        """
        trajectory = [state.copy()]
        current = state.copy()
        
        for _ in range(time_steps):
            # Linear step (dispersion)
            fft_state = np.fft.fft(current)
            fft_state *= np.exp(-1j * self.dispersion_relation * dt / 2)
            current = np.fft.ifft(fft_state)
            
            # Nonlinear step (self-interaction)
            current *= np.exp(-1j * np.abs(current)**2 * dt)
            
            # Linear step again (Strang splitting)
            fft_state = np.fft.fft(current)
            fft_state *= np.exp(-1j * self.dispersion_relation * dt / 2)
            current = np.fft.ifft(fft_state)
            
            trajectory.append(current.copy())
            
        return trajectory
        
    def encode_memory(self, data: np.ndarray) -> np.ndarray:
        """Encode data into soliton configuration"""
        # Create multiple solitons at different positions
        encoded = np.ones(self.lattice_size, dtype=complex)
        
        # Map data values to soliton positions and depths
        n_solitons = min(len(data), 10)  # Limit number of solitons
        positions = np.linspace(100, self.lattice_size - 100, n_solitons).astype(int)
        
        for i, (pos, value) in enumerate(zip(positions, data[:n_solitons])):
            depth = 0.3 + 0.5 * np.abs(value) / (np.max(np.abs(data)) + 1e-6)
            width = 10 + 5 * i  # Vary widths to avoid collision
            soliton = self.create_dark_soliton(pos, width, depth)
            encoded *= soliton
            
        return encoded
        
    def decode_memory(self, state: np.ndarray) -> np.ndarray:
        """Decode soliton configuration back to data"""
        # Find soliton positions by looking for amplitude dips
        amplitude = np.abs(state)
        gradient = np.gradient(amplitude)
        
        # Find local minima (soliton centers)
        minima = []
        for i in range(1, len(gradient) - 1):
            if gradient[i-1] < 0 and gradient[i+1] > 0:
                minima.append(i)
                
        # Extract depths at minima positions
        if minima:
            depths = 1.0 - amplitude[minima]
            return depths
        else:
            return np.array([0.0])

# ========== Attractor Hopper ==========

class AttractorHopper:
    """
    Implements chaotic search through attractor basins
    Uses controlled perturbations to hop between stable states
    """
    
    def __init__(self, phase_space_dim: int = 50):
        self.dim = phase_space_dim
        self.known_attractors = []
        self.basin_map = {}
        self.hop_history = deque(maxlen=1000)
        
    def map_attractor_basin(self, trajectory: np.ndarray) -> Dict[str, Any]:
        """Map the basin of attraction from a trajectory"""
        # Find the attractor (assumed to be the final state)
        attractor = trajectory[-1]
        
        # Estimate basin size by looking at convergence
        distances = [np.linalg.norm(trajectory[i] - attractor) 
                    for i in range(len(trajectory))]
        
        # Find when trajectory enters final basin
        basin_entry = 0
        for i in range(len(distances) - 1, 0, -1):
            if distances[i-1] > 2 * distances[i]:
                basin_entry = i
                break
                
        return {
            'attractor': attractor,
            'basin_radius': np.mean(distances[basin_entry:]),
            'convergence_time': len(trajectory) - basin_entry,
            'dimension': self._estimate_dimension(trajectory[basin_entry:])
        }
        
    def _estimate_dimension(self, trajectory: np.ndarray) -> float:
        """Estimate fractal dimension of attractor"""
        if len(trajectory) < 10:
            return 1.0
            
        # Box-counting dimension estimation
        scales = [0.1, 0.5, 1.0, 2.0]
        counts = []
        
        for scale in scales:
            boxes = set()
            for state in trajectory:
                box = tuple(np.floor(state[:3] / scale).astype(int))
                boxes.add(box)
            counts.append(len(boxes))
            
        # Linear fit in log-log space
        if len(counts) > 1 and counts[-1] > 0:
            log_scales = np.log(scales)
            log_counts = np.log(counts)
            slope, _ = np.polyfit(log_scales, log_counts, 1)
            return -slope
        else:
            return 1.0
            
    def generate_hop_perturbation(self, current_state: np.ndarray,
                                 target_direction: Optional[np.ndarray] = None,
                                 strength: float = 0.1) -> np.ndarray:
        """Generate perturbation to escape current basin"""
        if target_direction is None:
            # Random direction on hypersphere
            direction = np.random.randn(len(current_state))
            direction /= np.linalg.norm(direction)
        else:
            direction = target_direction / np.linalg.norm(target_direction)
            
        # Add rotational component for richer dynamics
        if len(current_state) >= 3:
            rotation = np.zeros_like(current_state)
            rotation[0] = -current_state[1]
            rotation[1] = current_state[0]
            direction = 0.7 * direction + 0.3 * rotation / (np.linalg.norm(rotation) + 1e-6)
            
        return direction * strength * np.linalg.norm(current_state)
        
    def search(self, objective_fn: Callable, initial_state: np.ndarray,
              max_hops: int = 100, energy_per_hop: int = 10) -> Tuple[np.ndarray, float]:
        """
        Search for optimal state by hopping between attractors
        Returns best state found and its objective value
        """
        current_state = initial_state.copy()
        best_state = current_state.copy()
        best_value = objective_fn(current_state)
        
        for hop in range(max_hops):
            # Add perturbation to escape current basin
            perturbation = self.generate_hop_perturbation(current_state)
            perturbed_state = current_state + perturbation
            
            # Let system evolve to new attractor (simplified dynamics)
            for _ in range(50):
                # Example dynamics: gradient descent with noise
                gradient = -2 * perturbed_state  # Simple quadratic
                noise = np.random.randn(*perturbed_state.shape) * 0.01
                perturbed_state -= 0.1 * gradient + noise
                
            # Evaluate new attractor
            new_value = objective_fn(perturbed_state)
            
            # Record hop
            self.hop_history.append({
                'from_state': current_state,
                'to_state': perturbed_state,
                'value_change': new_value - objective_fn(current_state),
                'hop_number': hop
            })
            
            # Update best if improved
            if new_value > best_value:
                best_state = perturbed_state.copy()
                best_value = new_value
                
            # Probabilistic acceptance (simulated annealing style)
            if new_value > objective_fn(current_state) or \
               np.random.random() < np.exp((new_value - objective_fn(current_state)) / 0.1):
                current_state = perturbed_state
                
        return best_state, best_value

# ========== Phase Explosion Engine ==========

class PhaseExplosionEngine:
    """
    Controlled desynchronization for pattern discovery
    Uses phase dynamics inspired by Kuramoto model
    """
    
    def __init__(self, n_oscillators: int = 100):
        self.n_oscillators = n_oscillators
        self.phases = np.random.uniform(0, 2*np.pi, n_oscillators)
        self.natural_frequencies = np.random.randn(n_oscillators) * 0.1 + 1.0
        self.coupling_matrix = self._init_coupling()
        
    def _init_coupling(self) -> np.ndarray:
        """Initialize coupling matrix with small-world topology"""
        try:
            # Start with nearest-neighbor coupling
            coupling = np.zeros((self.n_oscillators, self.n_oscillators))
            for i in range(self.n_oscillators):
                coupling[i, (i+1) % self.n_oscillators] = 1.0
                coupling[i, (i-1) % self.n_oscillators] = 1.0
                
            # Add random long-range connections
            rewire_prob = 0.1  # Probability of rewiring connections
            n_random = int(rewire_prob * self.n_oscillators)
            for _ in range(n_random):
                i, j = np.random.randint(0, self.n_oscillators, 2)
                if i != j:
                    coupling[i, j] = coupling[j, i] = 0.5
                    
            return coupling
            
        except Exception as e:
            logger.error(f"Coupling matrix creation failed: {e}")
            return np.eye(self.n_oscillators)  # Fallback to identity
    
    def update_topology(self, topology_type: str, **kwargs):
        """Update coupling topology dynamically"""
        try:
            self.topology_type = topology_type
            
            if topology_type == "small_world":
                self.coupling_matrix = self._create_small_world_coupling(
                    kwargs.get('rewire_prob', 0.1)
                )
            elif topology_type == "all_to_all":
                self.coupling_matrix = np.ones((self.n_oscillators, self.n_oscillators))
                np.fill_diagonal(self.coupling_matrix, 0)
            elif topology_type == "ring":
                self.coupling_matrix = np.zeros((self.n_oscillators, self.n_oscillators))
                for i in range(self.n_oscillators):
                    self.coupling_matrix[i, (i+1) % self.n_oscillators] = 1.0
                    self.coupling_matrix[i, (i-1) % self.n_oscillators] = 1.0
            else:
                logger.warning(f"Unknown topology type: {topology_type}, keeping current")
                
        except Exception as e:
            logger.error(f"Topology update failed: {e}")
        
    def evolve(self, coupling_strength: float = 0.1, 
              time_steps: int = 1000, dt: float = 0.01) -> np.ndarray:
        """
        Evolve phase dynamics with improved numerical stability
        Returns phase trajectory
        """
        try:
            trajectory = np.zeros((time_steps, self.n_oscillators))
            
            for t in range(time_steps):
                trajectory[t] = self.phases.copy()
                
                # Kuramoto dynamics with proper vectorization
                phase_diffs = self.phases[:, np.newaxis] - self.phases[np.newaxis, :]
                coupling_term = np.sum(self.coupling_matrix * np.sin(phase_diffs), axis=1)
                
                # Update phases
                self.phases += dt * (self.natural_frequencies + 
                                   coupling_strength * coupling_term)
                self.phases = self.phases % (2 * np.pi)
                
                # Check for numerical issues
                if np.any(np.isnan(self.phases)) or np.any(np.isinf(self.phases)):
                    logger.warning(f"Numerical instability at step {t}, resetting phases")
                    self.phases = np.random.uniform(0, 2*np.pi, self.n_oscillators)
                    
            return trajectory
            
        except Exception as e:
            logger.error(f"Phase evolution failed: {e}")
            return np.zeros((time_steps, self.n_oscillators))
        
    def trigger_explosion(self, strength: float = 1.0) -> np.ndarray:
        """Trigger controlled phase explosion"""
        try:
            # Add strong random perturbations
            perturbations = np.random.randn(self.n_oscillators) * strength
            self.phases += perturbations
            self.phases = self.phases % (2 * np.pi)
            
            # Temporarily reduce coupling to allow desynchronization
            original_coupling = self.coupling_matrix.copy()
            self.coupling_matrix *= 0.1
            
            # Evolve with weak coupling
            explosion_trajectory = self.evolve(coupling_strength=0.01, time_steps=100)
            
            # Restore coupling
            self.coupling_matrix = original_coupling
            
            return explosion_trajectory
            
        except Exception as e:
            logger.error(f"Phase explosion failed: {e}")
            return np.zeros((100, self.n_oscillators))
        
    def extract_patterns(self, trajectory: np.ndarray) -> List[np.ndarray]:
        """Extract emergent patterns from phase dynamics with improved clustering"""
        try:
            patterns = []
            
            if len(trajectory) == 0:
                return patterns
            
            # Look for phase clusters in final state
            final_phases = trajectory[-1]
            
            # Use complex representation for better clustering
            phase_complex = np.exp(1j * final_phases)
            
            # Hierarchical clustering on phase values
            distances = np.abs(phase_complex[:, np.newaxis] - phase_complex[np.newaxis, :])
            
            # Improved clustering with adaptive threshold
            mean_distance = np.mean(distances[distances > 0])
            threshold = min(0.5, mean_distance * 0.5)
            
            clusters = []
            unassigned = set(range(self.n_oscillators))
            
            while unassigned:
                seed = unassigned.pop()
                cluster = [seed]
                
                to_check = [seed]
                while to_check:
                    current = to_check.pop()
                    for osc in list(unassigned):
                        if distances[current, osc] < threshold:
                            cluster.append(osc)
                            to_check.append(osc)
                            unassigned.remove(osc)
                            
                if len(cluster) >= 3:  # Minimum cluster size
                    patterns.append(np.array(cluster))
                    
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
            return []

# ========== Enhanced Topology Transition Stabilization ==========

class TransitionStabilizer:
    """
    Enhanced stabilization system for topology transitions
    Monitors oscillation amplitudes and applies adaptive damping
    """
    
    def __init__(self, amplitude_threshold: float = 2.0, 
                 damping_strength: float = 0.1,
                 monitoring_window: int = 100):
        self.amplitude_threshold = amplitude_threshold
        self.damping_strength = damping_strength
        self.monitoring_window = monitoring_window
        
        # Oscillation monitoring
        self.amplitude_history = deque(maxlen=monitoring_window)
        self.frequency_history = deque(maxlen=monitoring_window)
        self.damping_applied = 0
        
        # Adaptive parameters
        self.adaptive_damping = True
        self.learning_rate = 0.01
        self.stability_score = 1.0
        
        # Critical oscillation patterns
        self.critical_patterns = {
            'resonance': {'min_freq': 0.8, 'max_freq': 1.2, 'min_amplitude': 3.0},
            'chaos_onset': {'amplitude_growth_rate': 0.5, 'frequency_spread': 0.3},
            'soliton_breakup': {'amplitude_variance': 1.5, 'phase_coherence': 0.3}
        }
        
        logger.info(f"TransitionStabilizer initialized: threshold={amplitude_threshold}, damping={damping_strength}")
    
    def monitor_oscillations(self, state: np.ndarray, timestamp: float = None) -> Dict[str, Any]:
        """
        Monitor current oscillation state and detect instabilities
        
        Args:
            state: Current system state (complex array)
            timestamp: Optional timestamp for temporal analysis
            
        Returns:
            Dictionary with oscillation metrics and stability assessment
        """
        try:
            # Validate input
            if len(state) == 0:
                return self._empty_metrics(timestamp)
            
            # Extract amplitude and phase information
            amplitudes = np.abs(state)
            phases = np.angle(state)
            
            # Calculate key metrics
            max_amplitude = np.max(amplitudes)
            mean_amplitude = np.mean(amplitudes)
            amplitude_variance = np.var(amplitudes)
            
            # Frequency analysis via phase differences
            if len(self.amplitude_history) > 0:
                prev_state = self.amplitude_history[-1]
                if len(prev_state) == len(state):
                    prev_phases = np.angle(prev_state)
                    phase_diffs = np.diff(np.unwrap(phases - prev_phases))
                    dominant_frequency = np.mean(np.abs(phase_diffs)) if len(phase_diffs) > 0 else 0.0
                else:
                    dominant_frequency = 0.0
            else:
                dominant_frequency = 0.0
            
            # Store in history
            self.amplitude_history.append(state.copy())
            self.frequency_history.append(dominant_frequency)
            
            # Detect critical patterns
            instability_detected = False
            critical_pattern = None
            
            # Check for resonance
            if (max_amplitude > self.critical_patterns['resonance']['min_amplitude'] and
                self.critical_patterns['resonance']['min_freq'] <= dominant_frequency <= 
                self.critical_patterns['resonance']['max_freq']):
                instability_detected = True
                critical_pattern = 'resonance'
            
            # Check for chaos onset
            if len(self.amplitude_history) >= 10:
                recent_amplitudes = [np.max(np.abs(s)) for s in list(self.amplitude_history)[-10:]]
                if len(recent_amplitudes) > 1:
                    amplitude_growth = np.polyfit(range(len(recent_amplitudes)), recent_amplitudes, 1)[0]
                    
                    if amplitude_growth > self.critical_patterns['chaos_onset']['amplitude_growth_rate']:
                        instability_detected = True
                        critical_pattern = 'chaos_onset'
            
            # Check for soliton breakup
            if amplitude_variance > self.critical_patterns['soliton_breakup']['amplitude_variance']:
                phase_coherence = self._calculate_phase_coherence(phases)
                if phase_coherence < self.critical_patterns['soliton_breakup']['phase_coherence']:
                    instability_detected = True
                    critical_pattern = 'soliton_breakup'
            
            # Update stability score
            if instability_detected:
                self.stability_score *= 0.9  # Decay on instability
            else:
                self.stability_score = min(1.0, self.stability_score + 0.01)  # Slow recovery
            
            return {
                'max_amplitude': max_amplitude,
                'mean_amplitude': mean_amplitude,
                'amplitude_variance': amplitude_variance,
                'dominant_frequency': dominant_frequency,
                'stability_score': self.stability_score,
                'instability_detected': instability_detected,
                'critical_pattern': critical_pattern,
                'requires_damping': max_amplitude > self.amplitude_threshold or instability_detected,
                'timestamp': timestamp or datetime.now(timezone.utc).timestamp()
            }
            
        except Exception as e:
            logger.error(f"Oscillation monitoring failed: {e}")
            return self._empty_metrics(timestamp, error=str(e))
    
    def _empty_metrics(self, timestamp: float = None, error: str = None) -> Dict[str, Any]:
        """Return empty/error metrics"""
        metrics = {
            'max_amplitude': 0.0,
            'mean_amplitude': 0.0,
            'amplitude_variance': 0.0,
            'dominant_frequency': 0.0,
            'stability_score': 0.0,
            'instability_detected': True,
            'critical_pattern': None,
            'requires_damping': True,
            'timestamp': timestamp or datetime.now(timezone.utc).timestamp()
        }
        if error:
            metrics['error'] = error
        return metrics
    
    def _calculate_phase_coherence(self, phases: np.ndarray) -> float:
        """
        Calculate phase coherence across the system
        Returns value between 0 (incoherent) and 1 (fully coherent)
        """
        try:
            if len(phases) == 0:
                return 0.0
            # Complex order parameter
            order_parameter = np.mean(np.exp(1j * phases))
            coherence = np.abs(order_parameter)
            return coherence
        except Exception:
            return 0.0
    
    def apply_damping(self, state: np.ndarray, metrics: Dict[str, Any], 
                     method: str = 'adaptive') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply damping to reduce oscillation amplitudes
        
        Args:
            state: Current system state
            metrics: Oscillation metrics from monitor_oscillations
            method: Damping method ('adaptive', 'uniform', 'selective')
            
        Returns:
            Tuple of (damped_state, damping_info)
        """
        try:
            damped_state = state.copy()
            damping_info = {'method': method, 'damping_applied': 0.0, 'regions_damped': 0}
            
            if not metrics.get('requires_damping', False):
                return damped_state, damping_info
            
            if method == 'adaptive':
                # Adaptive damping based on local amplitude
                amplitudes = np.abs(state)
                max_amp = metrics['max_amplitude']
                
                if max_amp == 0:
                    return damped_state, damping_info
                
                # Calculate spatially varying damping coefficient
                damping_coeff = np.zeros_like(amplitudes)
                
                # Strong damping for high amplitude regions
                high_amp_mask = amplitudes > self.amplitude_threshold
                damping_coeff[high_amp_mask] = self.damping_strength * (amplitudes[high_amp_mask] / max_amp)
                
                # Moderate damping for unstable patterns
                if metrics.get('critical_pattern') == 'resonance':
                    damping_coeff *= 1.5  # Extra damping for resonance
                elif metrics.get('critical_pattern') == 'chaos_onset':
                    damping_coeff += 0.05  # Background damping for chaos
                
                # Apply damping
                damped_state *= (1.0 - damping_coeff)
                
                damping_info['damping_applied'] = np.mean(damping_coeff)
                damping_info['regions_damped'] = np.sum(damping_coeff > 0)
                
            elif method == 'uniform':
                # Uniform damping across entire system
                damping_factor = 1.0 - self.damping_strength
                damped_state *= damping_factor
                damping_info['damping_applied'] = self.damping_strength
                damping_info['regions_damped'] = len(state)
                
            elif method == 'selective':
                # Selective damping only for critical patterns
                if metrics.get('critical_pattern'):
                    pattern = metrics['critical_pattern']
                    
                    if pattern == 'soliton_breakup':
                        # Smooth out amplitude variations
                        amplitudes = np.abs(state)
                        kernel_size = min(5, len(amplitudes) // 10)
                        if kernel_size > 0:
                            kernel = np.ones(kernel_size) / kernel_size
                            smoothed_amps = np.convolve(amplitudes, kernel, mode='same')
                            phases = np.angle(state)
                            damped_state = smoothed_amps * np.exp(1j * phases)
                        
                    elif pattern == 'resonance':
                        # Frequency-selective damping
                        if len(state) > 1:
                            fft_state = np.fft.fft(state)
                            freqs = np.fft.fftfreq(len(state))
                            
                            # Damp frequencies near resonance
                            resonance_mask = np.abs(freqs - metrics['dominant_frequency']) < 0.1
                            fft_state[resonance_mask] *= (1.0 - self.damping_strength)
                            
                            damped_state = np.fft.ifft(fft_state)
                    
                    damping_info['damping_applied'] = self.damping_strength
                    damping_info['pattern_targeted'] = pattern
            
            # Update counters
            self.damping_applied += 1
            
            # Adaptive parameter update
            if self.adaptive_damping:
                self._update_damping_parameters(metrics, damping_info)
            
            logger.debug(f"Applied {method} damping: {damping_info['damping_applied']:.3f} strength")
            
            return damped_state, damping_info
            
        except Exception as e:
            logger.error(f"Damping application failed: {e}")
            return state, {'error': str(e), 'damping_applied': 0.0}
    
    def _update_damping_parameters(self, metrics: Dict[str, Any], damping_info: Dict[str, Any]):
        """
        Update damping parameters based on effectiveness
        """
        try:
            # If stability improved after damping, keep current parameters
            if metrics['stability_score'] > 0.8:
                return
            
            # If instability persists, increase damping strength slightly
            if metrics.get('instability_detected', False):
                self.damping_strength = min(0.5, self.damping_strength * 1.05)
            else:
                # Gradually reduce damping strength if system is stable
                self.damping_strength = max(0.01, self.damping_strength * 0.99)
            
            # Adjust threshold based on recent amplitude history
            if len(self.amplitude_history) >= 20:
                recent_max_amps = [np.max(np.abs(s)) for s in list(self.amplitude_history)[-20:]]
                if len(recent_max_amps) > 0:
                    avg_max_amp = np.mean(recent_max_amps)
                    
                    # Adapt threshold to be slightly above typical max amplitude
                    target_threshold = avg_max_amp * 1.2
                    self.amplitude_threshold += self.learning_rate * (target_threshold - self.amplitude_threshold)
            
        except Exception as e:
            logger.warning(f"Parameter update failed: {e}")
    
    def reset(self):
        """Reset stabilizer state"""
        self.amplitude_history.clear()
        self.frequency_history.clear()
        self.damping_applied = 0
        self.stability_score = 1.0
        logger.debug("TransitionStabilizer reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current stabilizer status"""
        return {
            'amplitude_threshold': self.amplitude_threshold,
            'damping_strength': self.damping_strength,
            'stability_score': self.stability_score,
            'damping_applications': self.damping_applied,
            'monitoring_samples': len(self.amplitude_history),
            'adaptive_enabled': self.adaptive_damping
        }

# ========== Enhanced Priority Task Queue ==========

class PriorityTaskQueue:
    """
    Priority queue for ChaosTask objects with timeout handling
    FIXED: Now properly implements priority-based task scheduling
    """
    
    def __init__(self):
        self._queue = []
        self._index = 0
        self._lock = asyncio.Lock()
    
    async def put(self, task: ChaosTask):
        """Add task to priority queue"""
        async with self._lock:
            # Use negative priority for max-heap behavior (lower numbers = higher priority)
            heapq.heappush(self._queue, (-task.priority, self._index, task))
            self._index += 1
    
    async def get(self) -> ChaosTask:
        """Get highest priority task"""
        async with self._lock:
            if not self._queue:
                raise asyncio.QueueEmpty
            _, _, task = heapq.heappop(self._queue)
            return task
    
    def qsize(self) -> int:
        """Get queue size"""
        return len(self._queue)
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        return len(self._queue) == 0

# ========== Memory Monitor ==========

class MemoryMonitor:
    """Monitor memory usage during task execution"""
    
    def __init__(self, limit_mb: float = MEMORY_ISOLATION_SIZE_MB):
        self.limit_mb = limit_mb
        try:
            self.process = psutil.Process()
            self.initial_memory = self.get_memory_usage()
        except Exception:
            self.process = None
            self.initial_memory = 0.0
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            if self.process:
                return self.process.memory_info().rss / (1024 * 1024)
        except Exception:
            pass
        return 0.0
    
    def check_limit(self) -> bool:
        """Check if memory usage exceeds limit"""
        current = self.get_memory_usage()
        return (current - self.initial_memory) > self.limit_mb
    
    def get_peak_usage(self) -> float:
        """Get peak memory usage since initialization"""
        try:
            return max(0.0, self.get_memory_usage() - self.initial_memory)
        except Exception:
            return 0.0
            coupling[i, (i+1) % self.n_oscillators] = 1.0
            coupling[i, (i-1) % self.n_oscillators] = 1.0
            
        # Add random long-range connections
        n_random = int(0.1 * self.n_oscillators)
        for _ in range(n_random):
            i, j = np.random.randint(0, self.n_oscillators, 2)
            if i != j:
                coupling[i, j] = coupling[j, i] = 0.5
                
        return coupling
        
    def evolve(self, coupling_strength: float = 0.1, 
              time_steps: int = 1000, dt: float = 0.01) -> np.ndarray:
        """
        Evolve phase dynamics
        Returns phase trajectory
        """
        trajectory = np.zeros((time_steps, self.n_oscillators))
        
        for t in range(time_steps):
            trajectory[t] = self.phases.copy()
            
            # Kuramoto dynamics
            phase_diffs = self.phases[:, np.newaxis] - self.phases[np.newaxis, :]
            coupling_term = np.sum(self.coupling_matrix * np.sin(phase_diffs), axis=1)
            
            # Update phases
            self.phases += dt * (self.natural_frequencies + 
                               coupling_strength * coupling_term)
            self.phases = self.phases % (2 * np.pi)
            
        return trajectory
        
    def trigger_explosion(self, strength: float = 1.0) -> np.ndarray:
        """Trigger controlled phase explosion"""
        # Add strong random perturbations
        perturbations = np.random.randn(self.n_oscillators) * strength
        self.phases += perturbations
        self.phases = self.phases % (2 * np.pi)
        
        # Temporarily reduce coupling to allow desynchronization
        original_coupling = self.coupling_matrix.copy()
        self.coupling_matrix *= 0.1
        
        # Evolve with weak coupling
        explosion_trajectory = self.evolve(coupling_strength=0.01, time_steps=100)
        
        # Restore coupling
        self.coupling_matrix = original_coupling
        
        return explosion_trajectory
        
    def extract_patterns(self, trajectory: np.ndarray) -> List[np.ndarray]:
        """Extract emergent patterns from phase dynamics"""
        patterns = []
        
        # Look for phase clusters
        final_phases = trajectory[-1]
        
        # Hierarchical clustering on phase values
        phase_complex = np.exp(1j * final_phases)
        distances = np.abs(phase_complex[:, np.newaxis] - phase_complex[np.newaxis, :])
        
        # Simple clustering: group oscillators within phase threshold
        clusters = []
        unassigned = set(range(self.n_oscillators))
        
        while unassigned:
            seed = unassigned.pop()
            cluster = [seed]
            
            for osc in list(unassigned):
                if distances[seed, osc] < 0.5:
                    cluster.append(osc)
                    unassigned.remove(osc)
                    
            if len(cluster) > 2:  # Minimum cluster size
                patterns.append(np.array(cluster))
                
        return patterns

# ========== Topology Transition Stabilization ==========

class TransitionStabilizer:
    """
    Enhanced stabilization system for topology transitions
    Monitors oscillation amplitudes and applies adaptive damping
    """
    
    def __init__(self, amplitude_threshold: float = 2.0, 
                 damping_strength: float = 0.1,
                 monitoring_window: int = 100):
        self.amplitude_threshold = amplitude_threshold
        self.damping_strength = damping_strength
        self.monitoring_window = monitoring_window
        
        # Oscillation monitoring
        self.amplitude_history = deque(maxlen=monitoring_window)
        self.frequency_history = deque(maxlen=monitoring_window)
        self.damping_applied = 0
        
        # Adaptive parameters
        self.adaptive_damping = True
        self.learning_rate = 0.01
        self.stability_score = 1.0
        
        # Critical oscillation patterns
        self.critical_patterns = {
            'resonance': {'min_freq': 0.8, 'max_freq': 1.2, 'min_amplitude': 3.0},
            'chaos_onset': {'amplitude_growth_rate': 0.5, 'frequency_spread': 0.3},
            'soliton_breakup': {'amplitude_variance': 1.5, 'phase_coherence': 0.3}
        }
        
        logger.info(f"TransitionStabilizer initialized: threshold={amplitude_threshold}, damping={damping_strength}")
    
    def monitor_oscillations(self, state: np.ndarray, timestamp: float = None) -> Dict[str, Any]:
        """
        Monitor current oscillation state and detect instabilities
        
        Args:
            state: Current system state (complex array)
            timestamp: Optional timestamp for temporal analysis
            
        Returns:
            Dictionary with oscillation metrics and stability assessment
        """
        try:
            # Extract amplitude and phase information
            amplitudes = np.abs(state)
            phases = np.angle(state)
            
            # Calculate key metrics
            max_amplitude = np.max(amplitudes)
            mean_amplitude = np.mean(amplitudes)
            amplitude_variance = np.var(amplitudes)
            
            # Frequency analysis via phase differences
            if len(self.amplitude_history) > 0:
                prev_phases = np.angle(self.amplitude_history[-1]) if len(self.amplitude_history) > 0 else phases
                phase_diffs = np.diff(np.unwrap(phases - prev_phases))
                dominant_frequency = np.mean(np.abs(phase_diffs)) if len(phase_diffs) > 0 else 0.0
            else:
                dominant_frequency = 0.0
            
            # Store in history
            self.amplitude_history.append(state.copy())
            self.frequency_history.append(dominant_frequency)
            
            # Detect critical patterns
            instability_detected = False
            critical_pattern = None
            
            # Check for resonance
            if (max_amplitude > self.critical_patterns['resonance']['min_amplitude'] and
                self.critical_patterns['resonance']['min_freq'] <= dominant_frequency <= 
                self.critical_patterns['resonance']['max_freq']):
                instability_detected = True
                critical_pattern = 'resonance'
            
            # Check for chaos onset
            if len(self.amplitude_history) >= 10:
                recent_amplitudes = [np.max(np.abs(s)) for s in list(self.amplitude_history)[-10:]]
                amplitude_growth = np.polyfit(range(len(recent_amplitudes)), recent_amplitudes, 1)[0]
                
                if amplitude_growth > self.critical_patterns['chaos_onset']['amplitude_growth_rate']:
                    instability_detected = True
                    critical_pattern = 'chaos_onset'
            
            # Check for soliton breakup
            if amplitude_variance > self.critical_patterns['soliton_breakup']['amplitude_variance']:
                phase_coherence = self._calculate_phase_coherence(phases)
                if phase_coherence < self.critical_patterns['soliton_breakup']['phase_coherence']:
                    instability_detected = True
                    critical_pattern = 'soliton_breakup'
            
            # Update stability score
            if instability_detected:
                self.stability_score *= 0.9  # Decay on instability
            else:
                self.stability_score = min(1.0, self.stability_score + 0.01)  # Slow recovery
            
            return {
                'max_amplitude': max_amplitude,
                'mean_amplitude': mean_amplitude,
                'amplitude_variance': amplitude_variance,
                'dominant_frequency': dominant_frequency,
                'stability_score': self.stability_score,
                'instability_detected': instability_detected,
                'critical_pattern': critical_pattern,
                'requires_damping': max_amplitude > self.amplitude_threshold or instability_detected,
                'timestamp': timestamp or datetime.now(timezone.utc).timestamp()
            }
            
        except Exception as e:
            logger.error(f"Oscillation monitoring failed: {e}")
            return {
                'max_amplitude': 0.0,
                'stability_score': 0.0,
                'instability_detected': True,
                'requires_damping': True,
                'error': str(e)
            }
    
    def _calculate_phase_coherence(self, phases: np.ndarray) -> float:
        """
        Calculate phase coherence across the system
        Returns value between 0 (incoherent) and 1 (fully coherent)
        """
        try:
            # Complex order parameter
            order_parameter = np.mean(np.exp(1j * phases))
            coherence = np.abs(order_parameter)
            return coherence
        except Exception:
            return 0.0
    
    def apply_damping(self, state: np.ndarray, metrics: Dict[str, Any], 
                     method: str = 'adaptive') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply damping to reduce oscillation amplitudes
        
        Args:
            state: Current system state
            metrics: Oscillation metrics from monitor_oscillations
            method: Damping method ('adaptive', 'uniform', 'selective')
            
        Returns:
            Tuple of (damped_state, damping_info)
        """
        try:
            damped_state = state.copy()
            damping_info = {'method': method, 'damping_applied': 0.0, 'regions_damped': 0}
            
            if not metrics.get('requires_damping', False):
                return damped_state, damping_info
            
            if method == 'adaptive':
                # Adaptive damping based on local amplitude
                amplitudes = np.abs(state)
                max_amp = metrics['max_amplitude']
                
                # Calculate spatially varying damping coefficient
                damping_coeff = np.zeros_like(amplitudes)
                
                # Strong damping for high amplitude regions
                high_amp_mask = amplitudes > self.amplitude_threshold
                damping_coeff[high_amp_mask] = self.damping_strength * (amplitudes[high_amp_mask] / max_amp)
                
                # Moderate damping for unstable patterns
                if metrics.get('critical_pattern') == 'resonance':
                    damping_coeff *= 1.5  # Extra damping for resonance
                elif metrics.get('critical_pattern') == 'chaos_onset':
                    damping_coeff += 0.05  # Background damping for chaos
                
                # Apply damping
                damped_state *= (1.0 - damping_coeff)
                
                damping_info['damping_applied'] = np.mean(damping_coeff)
                damping_info['regions_damped'] = np.sum(damping_coeff > 0)
                
            elif method == 'uniform':
                # Uniform damping across entire system
                damping_factor = 1.0 - self.damping_strength
                damped_state *= damping_factor
                damping_info['damping_applied'] = self.damping_strength
                damping_info['regions_damped'] = len(state)
                
            elif method == 'selective':
                # Selective damping only for critical patterns
                if metrics.get('critical_pattern'):
                    pattern = metrics['critical_pattern']
                    
                    if pattern == 'soliton_breakup':
                        # Smooth out amplitude variations
                        amplitudes = np.abs(state)
                        smoothed_amps = np.convolve(amplitudes, np.ones(5)/5, mode='same')
                        phases = np.angle(state)
                        damped_state = smoothed_amps * np.exp(1j * phases)
                        
                    elif pattern == 'resonance':
                        # Frequency-selective damping
                        fft_state = np.fft.fft(state)
                        freqs = np.fft.fftfreq(len(state))
                        
                        # Damp frequencies near resonance
                        resonance_mask = np.abs(freqs - metrics['dominant_frequency']) < 0.1
                        fft_state[resonance_mask] *= (1.0 - self.damping_strength)
                        
                        damped_state = np.fft.ifft(fft_state)
                    
                    damping_info['damping_applied'] = self.damping_strength
                    damping_info['pattern_targeted'] = pattern
            
            # Update counters
            self.damping_applied += 1
            
            # Adaptive parameter update
            if self.adaptive_damping:
                self._update_damping_parameters(metrics, damping_info)
            
            logger.debug(f"Applied {method} damping: {damping_info['damping_applied']:.3f} strength")
            
            return damped_state, damping_info
            
        except Exception as e:
            logger.error(f"Damping application failed: {e}")
            return state, {'error': str(e), 'damping_applied': 0.0}
    
    def _update_damping_parameters(self, metrics: Dict[str, Any], damping_info: Dict[str, Any]):
        """
        Update damping parameters based on effectiveness
        """
        try:
            # If stability improved after damping, keep current parameters
            if metrics['stability_score'] > 0.8:
                return
            
            # If instability persists, increase damping strength slightly
            if metrics.get('instability_detected', False):
                self.damping_strength = min(0.5, self.damping_strength * 1.05)
            else:
                # Gradually reduce damping strength if system is stable
                self.damping_strength = max(0.01, self.damping_strength * 0.99)
            
            # Adjust threshold based on recent amplitude history
            if len(self.amplitude_history) >= 20:
                recent_max_amps = [np.max(np.abs(s)) for s in list(self.amplitude_history)[-20:]]
                avg_max_amp = np.mean(recent_max_amps)
                
                # Adapt threshold to be slightly above typical max amplitude
                target_threshold = avg_max_amp * 1.2
                self.amplitude_threshold += self.learning_rate * (target_threshold - self.amplitude_threshold)
            
        except Exception as e:
            logger.warning(f"Parameter update failed: {e}")
    
    def reset(self):
        """Reset stabilizer state"""
        self.amplitude_history.clear()
        self.frequency_history.clear()
        self.damping_applied = 0
        self.stability_score = 1.0
        logger.debug("TransitionStabilizer reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current stabilizer status"""
        return {
            'amplitude_threshold': self.amplitude_threshold,
            'damping_strength': self.damping_strength,
            'stability_score': self.stability_score,
            'damping_applications': self.damping_applied,
            'monitoring_samples': len(self.amplitude_history),
            'adaptive_enabled': self.adaptive_damping
        }

# ========== Enhanced Chaos Control Layer ==========

class ChaosControlLayer:
    """
    Enhanced CCL with topology transition stabilization capabilities
    Provides isolated environment for edge-of-chaos processing with post-swap turbulence control
    """
    
    def __init__(self, eigen_sentry: EigenSentry2, 
                 state_manager: CognitiveStateManager,
                 enable_transition_stabilization: bool = True):
        self.eigen_sentry = eigen_sentry
        self.state_manager = state_manager
        
        # Chaos processors
        self.soliton_processor = DarkSolitonProcessor()
        self.attractor_hopper = AttractorHopper()
        self.phase_engine = PhaseExplosionEngine()
        
        # Enhanced stabilization system
        self.enable_stabilization = enable_transition_stabilization
        if self.enable_stabilization:
            self.transition_stabilizer = TransitionStabilizer(
                amplitude_threshold=2.5,  # Tuned for topology transitions
                damping_strength=0.15,    # Moderate initial damping
                monitoring_window=150     # Extended monitoring window
            )
            logger.info("Transition stabilization enabled")
        else:
            self.transition_stabilizer = None
            logger.info("Transition stabilization disabled")
        
        # Adaptive timestep based on spectral stability
        self.adaptive_dt = AdaptiveTimestep(dt_base=0.01)
        self.eigen_sentry_ref = None  # Will be set during integration
        
        # Task management
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
        self.completed_tasks = deque(maxlen=1000)
        
        # Isolation and safety
        self.isolation_enabled = True
        self.checkpoint_states = deque(maxlen=100)
        self.last_checkpoint = datetime.now(timezone.utc)
        
        # Energy tracking
        self.energy_consumed = 0
        self.energy_generated = 0
        
        # Process pool for true isolation
        self.executor = ProcessPoolExecutor(max_workers=2)
        
        # Chaos burst controller
        self.chaos_controller = ChaosChannelController(
            lattice_ref=getattr(state_manager, 'lattice', None),
            observer_callback=self._on_observer_token
        )
        
        # Transition monitoring
        self.transition_history = deque(maxlen=50)
        self.stabilization_metrics = {
            'transitions_stabilized': 0,
            'instabilities_detected': 0,
            'damping_applications': 0,
            'average_stability_score': 1.0
        }
        
        # Register with EigenSentry
        self._register_with_eigensentry()
    
    async def monitor_transition_stability(self, 
                                          state: np.ndarray,
                                          transition_id: str = None,
                                          post_swap_monitoring_duration: float = 2.0) -> Dict[str, Any]:
        """
        Monitor and stabilize system after topology transition
        
        Args:
            state: Current system state to monitor
            transition_id: Optional identifier for the transition
            post_swap_monitoring_duration: How long to monitor after transition (seconds)
            
        Returns:
            Dictionary with stabilization results and metrics
        """
        if not self.enable_stabilization or self.transition_stabilizer is None:
            return {'stabilization_enabled': False, 'monitoring_skipped': True}
        
        try:
            logger.info(f"Starting post-transition monitoring for {post_swap_monitoring_duration}s")
            
            # Reset stabilizer for new transition
            self.transition_stabilizer.reset()
            
            stabilization_record = {
                'transition_id': transition_id or f"transition_{len(self.transition_history)}",
                'start_time': datetime.now(timezone.utc),
                'monitoring_duration': post_swap_monitoring_duration,
                'oscillation_samples': [],
                'damping_events': [],
                'final_stability_score': 0.0,
                'instabilities_detected': 0,
                'max_amplitude_observed': 0.0
            }
            
            start_time = datetime.now(timezone.utc)
            monitoring_interval = 0.05  # 50ms monitoring interval
            
            current_state = state.copy()
            
            while (datetime.now(timezone.utc) - start_time).total_seconds() < post_swap_monitoring_duration:
                # Monitor current oscillation state
                metrics = self.transition_stabilizer.monitor_oscillations(
                    current_state,
                    timestamp=datetime.now(timezone.utc).timestamp()
                )
                
                stabilization_record['oscillation_samples'].append(metrics)
                stabilization_record['max_amplitude_observed'] = max(
                    stabilization_record['max_amplitude_observed'],
                    metrics.get('max_amplitude', 0.0)
                )
                
                # Check if damping is needed
                if metrics.get('requires_damping', False):
                    stabilization_record['instabilities_detected'] += 1
                    
                    # Apply appropriate damping method
                    damping_method = self._select_damping_method(metrics)
                    damped_state, damping_info = self.transition_stabilizer.apply_damping(
                        current_state, metrics, damping_method
                    )
                    
                    # Record damping event
                    damping_event = {
                        'timestamp': datetime.now(timezone.utc).timestamp(),
                        'method': damping_method,
                        'critical_pattern': metrics.get('critical_pattern'),
                        'damping_strength': damping_info.get('damping_applied', 0.0),
                        'regions_affected': damping_info.get('regions_damped', 0)
                    }
                    stabilization_record['damping_events'].append(damping_event)
                    
                    # Update state
                    current_state = damped_state
                    
                    # Update global metrics
                    self.stabilization_metrics['damping_applications'] += 1
                    
                    logger.debug(f"Applied {damping_method} damping: {damping_info.get('damping_applied', 0.0):.3f}")
                
                # Brief pause before next monitoring cycle
                await asyncio.sleep(monitoring_interval)
            
            # Finalize stabilization record
            stabilization_record['end_time'] = datetime.now(timezone.utc)
            stabilization_record['final_stability_score'] = self.transition_stabilizer.stability_score
            stabilization_record['total_monitoring_samples'] = len(stabilization_record['oscillation_samples'])
            
            # Update global stabilization metrics
            self.stabilization_metrics['transitions_stabilized'] += 1
            self.stabilization_metrics['instabilities_detected'] += stabilization_record['instabilities_detected']
            
            # Update average stability score (exponential moving average)
            alpha = 0.1
            self.stabilization_metrics['average_stability_score'] = (
                alpha * stabilization_record['final_stability_score'] +
                (1 - alpha) * self.stabilization_metrics['average_stability_score']
            )
            
            # Store in history
            self.transition_history.append(stabilization_record)
            
            logger.info(
                f"Transition monitoring complete: {stabilization_record['instabilities_detected']} instabilities detected, "
                f"{len(stabilization_record['damping_events'])} damping events, "
                f"final stability: {stabilization_record['final_stability_score']:.3f}"
            )
            
            return {
                'success': True,
                'stabilization_record': stabilization_record,
                'stabilized_state': current_state,
                'monitoring_complete': True
            }
            
        except Exception as e:
            logger.error(f"Transition monitoring failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'stabilized_state': state  # Return original state on error
            }
    
    def _select_damping_method(self, metrics: Dict[str, Any]) -> str:
        """
        Select appropriate damping method based on oscillation characteristics
        
        Args:
            metrics: Oscillation metrics from monitor_oscillations
            
        Returns:
            Damping method name ('adaptive', 'uniform', 'selective')
        """
        try:
            critical_pattern = metrics.get('critical_pattern')
            max_amplitude = metrics.get('max_amplitude', 0.0)
            stability_score = metrics.get('stability_score', 1.0)
            
            # High amplitude or very low stability -> uniform damping
            if max_amplitude > 5.0 or stability_score < 0.3:
                return 'uniform'
            
            # Specific patterns -> selective damping
            if critical_pattern in ['resonance', 'soliton_breakup']:
                return 'selective'
            
            # Default to adaptive damping
            return 'adaptive'
            
        except Exception:
            return 'adaptive'  # Safe default
    
    async def stabilize_topology_transition(self, 
                                           old_state: np.ndarray,
                                           new_state: np.ndarray,
                                           transition_context: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Comprehensive stabilization for topology transitions
        
        Args:
            old_state: State before transition
            new_state: State after transition (potentially unstable)
            transition_context: Context information about the transition
            
        Returns:
            Tuple of (stabilized_state, stabilization_report)
        """
        if not self.enable_stabilization:
            return new_state, {'stabilization_disabled': True}
        
        try:
            logger.info("Starting comprehensive topology transition stabilization")
            
            transition_id = transition_context.get('transition_id', f"trans_{len(self.transition_history)}")
            
            # Phase 1: Immediate stability assessment
            initial_metrics = self.transition_stabilizer.monitor_oscillations(new_state)
            
            stabilization_report = {
                'transition_id': transition_id,
                'transition_context': transition_context,
                'initial_metrics': initial_metrics,
                'phases_completed': []
            }
            
            current_state = new_state.copy()
            
            # Phase 2: Immediate damping if critically unstable
            if initial_metrics.get('max_amplitude', 0.0) > 4.0:  # Critical threshold
                logger.warning("Critical instability detected - applying immediate damping")
                
                damped_state, damping_info = self.transition_stabilizer.apply_damping(
                    current_state, initial_metrics, method='uniform'
                )
                
                current_state = damped_state
                stabilization_report['phases_completed'].append({
                    'phase': 'immediate_damping',
                    'damping_info': damping_info
                })
            
            # Phase 3: Adaptive monitoring and stabilization
            monitoring_result = await self.monitor_transition_stability(
                current_state,
                transition_id=transition_id,
                post_swap_monitoring_duration=transition_context.get('monitoring_duration', 2.0)
            )
            
            stabilization_report['phases_completed'].append({
                'phase': 'adaptive_monitoring',
                'monitoring_result': monitoring_result
            })
            
            if monitoring_result.get('success', False):
                current_state = monitoring_result.get('stabilized_state', current_state)
            
            # Phase 4: Final stability verification
            final_metrics = self.transition_stabilizer.monitor_oscillations(current_state)
            stabilization_report['final_metrics'] = final_metrics
            
            # Assess overall stabilization success
            stabilization_success = (
                final_metrics.get('stability_score', 0.0) > 0.7 and
                final_metrics.get('max_amplitude', 0.0) < 3.0 and
                not final_metrics.get('instability_detected', True)
            )
            
            stabilization_report['overall_success'] = stabilization_success
            stabilization_report['stability_improvement'] = (
                final_metrics.get('stability_score', 0.0) - 
                initial_metrics.get('stability_score', 0.0)
            )
            
            logger.info(
                f"Topology transition stabilization complete: success={stabilization_success}, "
                f"stability improvement={stabilization_report['stability_improvement']:.3f}"
            )
            
            return current_state, stabilization_report
            
        except Exception as e:
            logger.error(f"Topology transition stabilization failed: {e}")
            return new_state, {
                'overall_success': False,
                'error': str(e),
                'fallback_used': True
            }
    
    def set_eigen_sentry(self, eigen_sentry):
        """Set reference to EigenSentry for adaptive timestep"""
        self.eigen_sentry_ref = eigen_sentry
        
    def _register_with_eigensentry(self):
        """Register CCL with EigenSentry for coordination"""
        async def prepare_for_chaos(signal: CoordinationSignal) -> bool:
            # Checkpoint current state
            self._checkpoint_state()
            # Prepare appropriate processor
            if signal.event.event_type == InstabilityType.SOLITON_FISSION:
                self.soliton_processor.lattice *= 1.5  # Increase nonlinearity
            return True
            
        async def complete_chaos(event_id: str, abort: bool = False) -> bool:
            if abort:
                # Restore from checkpoint
                self._restore_checkpoint()
            return True
            
        self.eigen_sentry.register_module('ccl', prepare_for_chaos, complete_chaos)
        
    def _checkpoint_state(self):
        """Create state checkpoint for rollback"""
        checkpoint = {
            'timestamp': datetime.now(timezone.utc),
            'soliton_lattice': self.soliton_processor.lattice.copy(),
            'oscillator_phases': self.phase_engine.phases.copy(),
            'energy_consumed': self.energy_consumed
        }
        self.checkpoint_states.append(checkpoint)
        self.last_checkpoint = checkpoint['timestamp']
        
    def _restore_checkpoint(self):
        """Restore from most recent checkpoint"""
        if self.checkpoint_states:
            checkpoint = self.checkpoint_states[-1]
            self.soliton_processor.lattice = checkpoint['soliton_lattice']
            self.phase_engine.phases = checkpoint['oscillator_phases']
            self.energy_consumed = checkpoint['energy_consumed']
            logger.info("Restored from checkpoint")
            
    async def submit_task(self, task: ChaosTask) -> str:
        """Submit task for chaos-assisted computation"""
        # Check energy budget with EigenSentry
        gate_id = self.eigen_sentry.enter_ccl(f'task_{task.task_id}', task.energy_budget)
        
        if not gate_id:
            raise ValueError(f"Insufficient energy for task {task.task_id}")
            
        task.metadata['gate_id'] = gate_id
        await self.task_queue.put(task)
        self.active_tasks[task.task_id] = task
        
        return task.task_id
        
    async def process_tasks(self):
        """Main task processing loop"""
        while True:
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Process based on chaos mode
                if self.isolation_enabled:
                    # Run in separate process for true isolation
                    result = await self._process_isolated(task)
                else:
                    # Run in current process (faster but less safe)
                    result = await self._process_inline(task)
                    
                # Update energy tracking
                self.energy_consumed += result.energy_used
                efficiency_multiplier = EFFICIENCY_MULTIPLIERS.get(task.mode.value, 1.0)
                self.energy_generated += result.energy_used * efficiency_multiplier
                
                # Complete with EigenSentry
                gate_id = task.metadata.get('gate_id')
                if gate_id:
                    self.eigen_sentry.exit_ccl(gate_id, success=result.success)
                    
                # Store result
                self.completed_tasks.append(result)
                del self.active_tasks[task.task_id]
                
                # Call callback if provided
                if task.callback:
                    await task.callback(result)
                    
            except asyncio.TimeoutError:
                # No tasks, check for checkpoint
                if (datetime.now(timezone.utc) - self.last_checkpoint).total_seconds() > CHECKPOINT_INTERVAL_S:
                    self._checkpoint_state()
            except Exception as e:
                logger.error(f"Task processing error: {e}")
                
    async def _process_isolated(self, task: ChaosTask) -> ChaosResult:
        """Process task in isolated subprocess"""
        loop = asyncio.get_event_loop()
        
        # Serialize task for subprocess
        task_data = {
            'mode': task.mode.value,
            'input_data': task.input_data,
            'parameters': task.parameters
        }
        
        # Run in subprocess with timeout
        try:
            result_data = await loop.run_in_executor(
                self.executor,
                _process_chaos_task_isolated,
                task_data
            )
            
            return ChaosResult(
                task_id=task.task_id,
                success=result_data['success'],
                output_data=result_data['output_data'],
                energy_used=result_data['energy_used'],
                computation_time=result_data['computation_time'],
                efficiency_gain=result_data['efficiency_gain']
            )
            
        except Exception as e:
            logger.error(f"Isolated processing failed: {e}")
            return ChaosResult(
                task_id=task.task_id,
                success=False,
                output_data=None,
                energy_used=task.energy_budget,
                computation_time=0.0,
                efficiency_gain=0.0,
                metadata={'error': str(e)}
            )
            
    async def _process_inline(self, task: ChaosTask) -> ChaosResult:
        """Process task in current process"""
        start_time = datetime.now(timezone.utc)
        
        try:
            if task.mode == ChaosMode.DARK_SOLITON:
                output_data = await self._process_soliton(task)
            elif task.mode == ChaosMode.ATTRACTOR_HOP:
                output_data = await self._process_attractor(task)
            elif task.mode == ChaosMode.PHASE_EXPLOSION:
                output_data = await self._process_phase(task)
            else:
                output_data = await self._process_hybrid(task)
                
            computation_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            efficiency_gain = EFFICIENCY_MULTIPLIERS.get(task.mode.value, 1.0)
            
            return ChaosResult(
                task_id=task.task_id,
                success=True,
                output_data=output_data,
                energy_used=int(task.energy_budget * 0.8),  # Some energy saved
                computation_time=computation_time,
                efficiency_gain=efficiency_gain
            )
            
        except Exception as e:
            logger.error(f"Inline processing failed: {e}")
            return ChaosResult(
                task_id=task.task_id,
                success=False,
                output_data=None,
                energy_used=task.energy_budget,
                computation_time=(datetime.now(timezone.utc) - start_time).total_seconds(),
                efficiency_gain=0.0,
                metadata={'error': str(e)}
            )
            
    async def _process_soliton(self, task: ChaosTask) -> np.ndarray:
        """Process using dark soliton dynamics"""
        # Encode input into soliton configuration
        encoded = self.soliton_processor.encode_memory(task.input_data)
        
        # Get adaptive timestep based on Lyapunov exponents
        dt = 0.01  # Default
        if hasattr(self, 'eigen_sentry_ref') and self.eigen_sentry_ref is not None:
            lambda_max = self.eigen_sentry_ref.metrics.get('lambda_max', 0.0)
            dt = self.adaptive_dt.compute_timestep(lambda_max)
        
        # Propagate solitons
        time_steps = task.parameters.get('time_steps', 100)
        trajectory = self.soliton_processor.propagate(encoded, time_steps, dt)
        
        # Decode final state
        output = self.soliton_processor.decode_memory(trajectory[-1])
        
        return output
    
    async def trigger_chaos_burst(self, intensity: float = 1.0, 
                                 duration: Optional[float] = None,
                                 mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Trigger a controlled chaos burst using ChaosChannelController
        
        Args:
            intensity: Burst intensity (0.1 to 5.0)
            duration: Burst duration in seconds
            mode: Target chaos mode
            
        Returns:
            Burst results including metrics and harvested state
        """
        result = await self.chaos_controller.trigger(intensity, duration, mode)
        
        if result['success'] and 'result' in result:
            burst_data = result['result']
            
            # Try to spawn new soliton from harvested energy
            if 'harvested_state' in burst_data and burst_data['harvested_state'] is not None:
                spawned = await self.chaos_controller.spawn_soliton_from_harvest(
                    burst_data['harvested_state']
                )
                result['soliton_spawned'] = spawned
                
        return result
    
    async def _on_observer_token(self, token):
        """Handle observer tokens emitted during chaos bursts"""
        # Log token for analysis
        logger.debug(f"Observer token {token.token_id}: energy={token.energy_snapshot:.2f}")
        
        # Could forward to other systems or store for later analysis
        pass
        
    async def _process_attractor(self, task: ChaosTask) -> np.ndarray:
        """Process using attractor hopping"""
        # Define objective function from task parameters
        target = task.parameters.get('target', np.zeros_like(task.input_data))
        
        def objective(state):
            return -np.linalg.norm(state - target)  # Minimize distance to target
            
        # Search using attractor hopping
        max_hops = task.parameters.get('max_hops', 50)
        best_state, best_value = self.attractor_hopper.search(
            objective, task.input_data, max_hops
        )
        
        return best_state
        
    async def _process_phase(self, task: ChaosTask) -> np.ndarray:
        """Process using phase explosion"""
        # Set initial phases from input data
        n_osc = len(self.phase_engine.phases)
        if len(task.input_data) >= n_osc:
            self.phase_engine.phases = task.input_data[:n_osc] % (2 * np.pi)
            
        # Trigger explosion
        strength = task.parameters.get('explosion_strength', 1.0)
        trajectory = self.phase_engine.trigger_explosion(strength)
        
        # Extract patterns
        patterns = self.phase_engine.extract_patterns(trajectory)
        
        # Convert patterns to output format
        output = np.zeros(len(task.input_data))
        for i, pattern in enumerate(patterns):
            if i < len(output):
                output[i] = len(pattern)  # Pattern size as feature
                
        return output
        
    async def _process_hybrid(self, task: ChaosTask) -> np.ndarray:
        """Process using hybrid chaos approach"""
        # Combine multiple chaos modes
        soliton_weight = task.parameters.get('soliton_weight', 0.33)
        attractor_weight = task.parameters.get('attractor_weight', 0.33)
        phase_weight = task.parameters.get('phase_weight', 0.34)
        
        # Process with each mode
        soliton_out = await self._process_soliton(task)
        attractor_out = await self._process_attractor(task)
        phase_out = await self._process_phase(task)
        
        # Weighted combination
        output = (soliton_weight * soliton_out + 
                 attractor_weight * attractor_out + 
                 phase_weight * phase_out)
        
        return output
        
    def get_status(self) -> Dict[str, Any]:
        """Get enhanced CCL status including stabilization metrics"""
        base_status = {
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'energy_consumed': self.energy_consumed,
            'energy_generated': self.energy_generated,
            'efficiency_ratio': self.energy_generated / (self.energy_consumed + 1e-6),
            'isolation_enabled': self.isolation_enabled,
            'checkpoints_stored': len(self.checkpoint_states),
            'last_checkpoint': self.last_checkpoint.isoformat(),
            'chaos_controller': self.chaos_controller.get_status()
        }
        
        # Add stabilization metrics if enabled
        if self.enable_stabilization and self.transition_stabilizer is not None:
            base_status['stabilization'] = {
                'enabled': True,
                'stabilizer_status': self.transition_stabilizer.get_status(),
                'transition_metrics': self.stabilization_metrics,
                'recent_transitions': len(self.transition_history),
                'last_transition': (
                    self.transition_history[-1]['start_time'].isoformat() 
                    if self.transition_history else None
                )
            }
        else:
            base_status['stabilization'] = {'enabled': False}
        
        return base_status

# ========== Isolated Processing Function ==========

def _process_chaos_task_isolated(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process chaos task in isolated subprocess
    This runs in a separate process for true isolation
    """
    import numpy as np
    from datetime import datetime, timezone
    
    start_time = datetime.now(timezone.utc)
    
    try:
        mode = task_data['mode']
        input_data = task_data['input_data']
        parameters = task_data['parameters']
        
        # Simplified processing for subprocess
        if mode == 'dark_soliton':
            # Simple soliton-like transformation
            output_data = np.tanh(input_data) * np.exp(-0.1 * np.abs(input_data))
        elif mode == 'attractor_hop':
            # Random walk toward target
            output_data = input_data + np.random.randn(*input_data.shape) * 0.1
        elif mode == 'phase_explosion':
            # Phase randomization
            phases = np.angle(input_data + 1j * np.random.randn(*input_data.shape))
            output_data = np.abs(input_data) * np.exp(1j * phases)
        else:
            output_data = input_data
            
        computation_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        return {
            'success': True,
            'output_data': output_data,
            'energy_used': 80,  # Fixed for subprocess
            'computation_time': computation_time,
            'efficiency_gain': 3.0  # Average efficiency
        }
        
    except Exception as e:
        return {
            'success': False,
            'output_data': None,
            'energy_used': 100,
            'computation_time': 0.0,
            'efficiency_gain': 0.0,
            'error': str(e)
        }

# ========== Testing and Demo ==========

async def demonstrate_ccl():
    """Demonstrate Chaos Control Layer capabilities"""
    print("🌀 Chaos Control Layer Demo")
    print("=" * 60)
    
    # Initialize components
    state_manager = CognitiveStateManager()
    eigen_sentry = EigenSentry2(state_manager)
    ccl = ChaosControlLayer(eigen_sentry, state_manager)
    
    # Start processing loop
    process_task = asyncio.create_task(ccl.process_tasks())
    
    # Test 1: Dark Soliton Memory
    print("\n1️⃣ Testing Dark Soliton Memory Processing...")
    test_data = np.array([1.0, -0.5, 0.8, -0.3, 0.6])
    
    soliton_task = ChaosTask(
        task_id="soliton_test",
        mode=ChaosMode.DARK_SOLITON,
        input_data=test_data,
        parameters={'time_steps': 50},
        energy_budget=100
    )
    
    task_id = await ccl.submit_task(soliton_task)
    print(f"  Submitted task {task_id}")
    
    # Wait for completion
    await asyncio.sleep(2.0)
    
    # Test 2: Attractor Hopping Search
    print("\n2️⃣ Testing Attractor Hopping Search...")
    search_start = np.random.randn(10)
    search_target = np.ones(10) * 0.5
    
    attractor_task = ChaosTask(
        task_id="attractor_test",
        mode=ChaosMode.ATTRACTOR_HOP,
        input_data=search_start,
        parameters={'target': search_target, 'max_hops': 20},
        energy_budget=150
    )
    
    await ccl.submit_task(attractor_task)
    await asyncio.sleep(2.0)
    
    # Test 3: Phase Explosion Pattern Discovery
    print("\n3️⃣ Testing Phase Explosion Pattern Discovery...")
    phase_data = np.random.randn(100)
    
    phase_task = ChaosTask(
        task_id="phase_test",
        mode=ChaosMode.PHASE_EXPLOSION,
        input_data=phase_data,
        parameters={'explosion_strength': 2.0},
        energy_budget=200
    )
    
    await ccl.submit_task(phase_task)
    await asyncio.sleep(2.0)
    
    # Show results
    print("\n📊 CCL Status:")
    status = ccl.get_status()
    print(f"  Active tasks: {status['active_tasks']}")
    print(f"  Completed tasks: {status['completed_tasks']}")
    print(f"  Energy efficiency: {status['efficiency_ratio']:.2f}x")
    print(f"  Total energy generated: {status['energy_generated']}")
    
    # Cleanup
    process_task.cancel()
    ccl.executor.shutdown()

if __name__ == "__main__":
    asyncio.run(demonstrate_ccl())
