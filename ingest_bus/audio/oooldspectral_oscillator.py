"""
Spectral Koopman-Kuramoto Oscillator Network
Maps audio features to collective phase dynamics for holographic visualization
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from functools import lru_cache
import threading
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class OscillatorState:
    """Complete state of the oscillator network"""
    phases: np.ndarray          # Individual oscillator phases [0, 2π)
    natural_freqs: np.ndarray   # Natural frequencies ωᵢ (rad/s)
    coupling: float             # Coupling strength K
    coherence: float            # Order parameter magnitude r ∈ [0,1]
    collective_phase: float     # Collective phase ψ ∈ [0,2π)
    
class BanksyOscillator:
    """
    Kuramoto-style oscillator network driven by audio features.
    
    The collective behavior emerges from:
    - Natural frequencies centered on spectral centroid
    - Coupling strength modulated by emotion intensity
    - Phase dynamics following Kuramoto model
    """
    
    def __init__(self, 
                 n_osc: int = 12,
                 bandwidth: float = 100.0,
                 dt: float = 0.1,
                 noise_strength: float = 0.01):
        """
        Initialize oscillator network.
        
        Args:
            n_osc: Number of oscillators
            bandwidth: Frequency spread around centroid (Hz)
            dt: Integration timestep (seconds)
            noise_strength: Phase noise for realistic dynamics
        """
        self.n = n_osc
        self.bandwidth = bandwidth
        self.dt = dt
        self.noise_strength = noise_strength
        
        # Initialize phases randomly
        self.phases = np.random.uniform(0, 2*np.pi, self.n)
        
        # Natural frequencies (will be set by map_parameters)
        self.natural_freqs = np.zeros(self.n)
        
        # Coupling parameters
        self.K = 0.1  # Initial weak coupling
        self.K_min = 0.1
        self.K_max = 1.0
        
        # History for smoothing using deque for efficiency
        self.coherence_history = deque(maxlen=10)
        self.phase_history = deque(maxlen=5)
        self.freq_history = deque(maxlen=5)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance caching
        self._cache = {}
        self._cache_valid = False
        
        # Adaptation parameters
        self.freq_adaptation_rate = 0.1
        self.target_freqs = np.zeros(self.n)
        
        logger.info(f"Initialized BanksyOscillator with {n_osc} oscillators")
    
    def map_parameters(self, 
                      centroid: float, 
                      emotion_intensity: float,
                      rms: Optional[float] = None):
        with self._lock:  # Thread-safe parameter mapping
            self._map_parameters_unsafe(centroid, emotion_intensity, rms)
    
    def _map_parameters_unsafe(self, 
                              centroid: float, 
                              emotion_intensity: float,
                              rms: Optional[float] = None):
        """
        Map audio features to oscillator parameters.
        
        Args:
            centroid: Spectral centroid in Hz
            emotion_intensity: Emotion strength ∈ [0,1]
            rms: Optional RMS amplitude for additional modulation
        """
        # 1. Natural frequencies centered on spectral centroid
        # Create frequency distribution around centroid
        if self.n == 1:
            deltas = np.array([0.0])
        else:
            deltas = np.linspace(-self.bandwidth, self.bandwidth, self.n)
        
        # Convert to rad/s and add slight randomness for realism
        self.target_freqs = 2 * np.pi * (centroid + deltas)
        
        # Smooth frequency adaptation
        self.natural_freqs = (
            (1 - self.freq_adaptation_rate) * self.natural_freqs +
            self.freq_adaptation_rate * self.target_freqs
        )
        
        # 2. Coupling strength from emotion intensity
        # Higher emotion → stronger coupling → more synchronization
        self.K = self.K_min + emotion_intensity * (self.K_max - self.K_min)
        
        # 3. Optional RMS modulation
        if rms is not None:
            # Boost coupling for louder sounds
            amplitude_boost = np.clip(rms * 2, 0, 1)
            self.K *= (1 + 0.5 * amplitude_boost)
        
        # Store in history for interpolation
        self.freq_history.append(self.natural_freqs.copy())
        
        # Invalidate cache
        self._cache_valid = False
            
        logger.debug(f"Mapped: f0={centroid:.1f}Hz, K={self.K:.3f}")
    
    def step(self, dt: Optional[float] = None):
        """
        Advance oscillator phases using Kuramoto dynamics.
        
        The Kuramoto model:
        dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ) + ξᵢ
        
        where ξᵢ is optional noise for realistic behavior.
        """
        if dt is None:
            dt = self.dt
            
        # Compute pairwise phase differences
        # phase_diffs[i,j] = sin(θⱼ - θᵢ)
        phase_diffs = np.sin(self.phases[None, :] - self.phases[:, None])
        
        # Kuramoto coupling term
        coupling_term = (self.K / self.n) * phase_diffs.sum(axis=1)
        
        # Phase velocity with optional noise
        noise = self.noise_strength * np.random.randn(self.n)
        dtheta = self.natural_freqs + coupling_term + noise
        
        # Update phases (wrapped to [0, 2π))
        self.phases = (self.phases + dtheta * dt) % (2 * np.pi)
    
    def compute_order_parameter(self) -> Tuple[float, float]:
        """
        Compute Kuramoto order parameter (r, ψ).
        
        r * e^(iψ) = (1/N) Σⱼ e^(iθⱼ)
        
        Returns:
            (r, ψ): coherence and collective phase
        """
        # Complex order parameter
        order = np.mean(np.exp(1j * self.phases))
        
        r = np.abs(order)
        psi = np.angle(order) % (2 * np.pi)
        
        # Smooth coherence for stability
        self.coherence_history.append(r)
        r_smooth = np.mean(list(self.coherence_history)) if self.coherence_history else r
        
        return r_smooth, psi
    
    def psi_state(self) -> Dict[str, any]:
        """
        Get complete oscillator state for downstream processing.
        
        Returns:
            Dictionary with coherence, collective phase, and individual phases
        """
        r, psi = self.compute_order_parameter()
        
        state = {
            "phase_coherence": float(r),      # Use existing schema field name
            "psi_phase": float(psi),          # Use existing schema field name
            "psi_magnitude": float(r),        # Duplicate for compatibility
            "oscillator_phases": self.phases.tolist(),
            "oscillator_frequencies": (self.natural_freqs / (2 * np.pi)).tolist(),  # Hz
            "coupling_strength": float(self.K),
            "dominant_frequency": float(np.mean(self.natural_freqs) / (2 * np.pi))
        }
        
        return state
    
    def get_full_state(self) -> OscillatorState:
        """Get complete state as dataclass"""
        r, psi = self.compute_order_parameter()
        
        return OscillatorState(
            phases=self.phases.copy(),
            natural_freqs=self.natural_freqs.copy(),
            coupling=self.K,
            coherence=r,
            collective_phase=psi
        )
    
    def reset(self):
        """Reset oscillator to initial conditions"""
        with self._lock:
            self.phases = np.random.uniform(0, 2*np.pi, self.n)
            self.natural_freqs = np.zeros(self.n)
            self.K = self.K_min
            self.coherence_history.clear()
            self.phase_history.clear()
            self.freq_history.clear()
            self._cache.clear()
            self._cache_valid = False
            logger.info("Oscillator reset")
    
    def set_phases(self, phases: List[float]):
        """Manually set oscillator phases (for testing)"""
        if len(phases) != self.n:
            raise ValueError(f"Expected {self.n} phases, got {len(phases)}")
        self.phases = np.array(phases) % (2 * np.pi)
    
    def get_wavefield_params(self) -> Dict[str, any]:
        """Get parameters for holographic wavefield modulation"""
        r, psi = self.compute_order_parameter()
        return {
            'phase_modulation': float(psi),
            'coherence': float(r),
            'oscillator_phases': self.phases.tolist(),
            'dominant_freq': float(np.mean(self.natural_freqs) / (2 * np.pi)),
            'spatial_frequencies': self._compute_spatial_frequencies()
        }
    
    def _compute_spatial_frequencies(self) -> List[Tuple[float, float]]:
        """Map oscillator frequencies to spatial wave vectors for hologram"""
        # Vectorized computation for better performance
        angles = np.arange(self.n) * np.pi / self.n
        freq_normalized = self.natural_freqs / (2 * np.pi * 1000)
        
        # Create spiral arrangement for better spatial distribution
        spiral_factor = 1 + 0.1 * np.arange(self.n)
        kx = freq_normalized * np.cos(angles) * spiral_factor
        ky = freq_normalized * np.sin(angles) * spiral_factor
        
        return list(zip(kx.astype(float), ky.astype(float)))
    
    def compute_wavefield_interference(self, grid_size: int = 128) -> np.ndarray:
        """Compute holographic interference pattern from oscillator network"""
        # Create spatial grid
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Initialize complex wavefield
        wavefield = np.zeros((grid_size, grid_size), dtype=complex)
        
        # Get spatial frequencies and phases
        spatial_freqs = self._compute_spatial_frequencies()
        
        # Each oscillator contributes a plane wave
        for i, (kx, ky) in enumerate(spatial_freqs):
            phase = self.phases[i]
            amplitude = 1.0 / np.sqrt(self.n)  # Normalize
            
            # Add plane wave contribution
            wavefield += amplitude * np.exp(1j * (kx * X + ky * Y + phase))
        
        # Include coherence modulation
        r, psi = self.compute_order_parameter()
        wavefield *= np.exp(1j * psi) * (0.5 + 0.5 * r)
        
        return wavefield
    
    def get_phase_gradients(self) -> Dict[str, np.ndarray]:
        """Compute phase gradients for wavefront reconstruction"""
        # Sort oscillators by phase for gradient computation
        sorted_indices = np.argsort(self.phases)
        sorted_phases = self.phases[sorted_indices]
        sorted_freqs = self.natural_freqs[sorted_indices]
        
        # Compute phase differences (wrapped)
        phase_diffs = np.diff(sorted_phases)
        phase_diffs = (phase_diffs + np.pi) % (2 * np.pi) - np.pi
        
        # Frequency gradients
        freq_grads = np.diff(sorted_freqs)
        
        return {
            'phase_gradients': phase_diffs,
            'frequency_gradients': freq_grads,
            'gradient_indices': sorted_indices,
            'mean_gradient': float(np.mean(np.abs(phase_diffs)))
        }
    
    @lru_cache(maxsize=32)
    def _compute_wavefield_basis(self, grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Cached computation of spatial grid for wavefield"""
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        return np.meshgrid(x, y)
    
    def compute_multiscale_wavefield(self, scales: List[int] = None) -> Dict[str, np.ndarray]:
        """Compute wavefield at multiple spatial scales"""
        if scales is None:
            scales = [64, 128, 256]
        
        wavefields = {}
        for scale in scales:
            wavefields[f'scale_{scale}'] = self.compute_wavefield_interference(scale)
        
        return wavefields
    
    def get_adaptive_spatial_mapping(self, 
                                   audio_bandwidth: float,
                                   spatial_extent: float = 2.0) -> Dict[str, Any]:
        """Adaptively map audio frequencies to spatial domain"""
        # Compute optimal spatial frequency scaling
        freq_range = np.ptp(self.natural_freqs) / (2 * np.pi)
        scale_factor = spatial_extent / (freq_range / 1000 + 1e-6)
        
        # Compute adaptive spiral parameters
        coherence = self.coherence_history[-1] if self.coherence_history else 0.5
        spiral_tightness = 0.05 + 0.15 * (1 - coherence)
        
        # Generate adaptive spatial frequencies
        angles = np.arange(self.n) * 2 * np.pi / self.n
        radii = scale_factor * self.natural_freqs / (2 * np.pi * 1000)
        
        # Apply logarithmic spiral for better distribution
        spiral_angles = angles + spiral_tightness * np.log(radii + 1)
        
        kx = radii * np.cos(spiral_angles)
        ky = radii * np.sin(spiral_angles)
        
        return {
            'spatial_frequencies': list(zip(kx.astype(float), ky.astype(float))),
            'scale_factor': float(scale_factor),
            'spiral_tightness': float(spiral_tightness),
            'frequency_range': float(freq_range),
            'coherence_modulation': float(coherence)
        }
    
    def interpolate_parameters(self, alpha: float) -> Dict[str, Any]:
        """Interpolate between current and previous parameters"""
        if len(self.freq_history) < 2:
            return self.psi_state()
        
        with self._lock:
            # Get current and previous frequencies
            freq_current = self.freq_history[-1]
            freq_prev = self.freq_history[-2]
            
            # Smooth interpolation
            freq_interp = (1 - alpha) * freq_prev + alpha * freq_current
            
            # Temporarily update frequencies for computation
            original_freqs = self.natural_freqs.copy()
            self.natural_freqs = freq_interp
            
            # Compute interpolated state
            state = self.psi_state()
            
            # Restore original frequencies
            self.natural_freqs = original_freqs
            
            return state
    
    def compute_phase_velocity_field(self, grid_size: int = 64) -> np.ndarray:
        """Compute instantaneous phase velocity field"""
        X, Y = self._compute_wavefield_basis(grid_size)
        
        # Initialize velocity field
        vx = np.zeros((grid_size, grid_size))
        vy = np.zeros((grid_size, grid_size))
        
        spatial_freqs = self._compute_spatial_frequencies()
        
        for i, (kx, ky) in enumerate(spatial_freqs):
            # Phase velocity = ω/k
            omega = self.natural_freqs[i]
            k_mag = np.sqrt(kx**2 + ky**2) + 1e-6
            
            # Contribution weighted by oscillator amplitude
            weight = 1.0 / self.n
            vx += weight * omega * kx / k_mag**2
            vy += weight * omega * ky / k_mag**2
        
        # Modulate by coherence
        r, _ = self.compute_order_parameter()
        velocity_field = np.stack([vx, vy]) * (0.5 + 0.5 * r)
        
        return velocity_field

# Specialized variants for different audio contexts

class VoiceOscillator(BanksyOscillator):
    """Oscillator tuned for voice/speech processing"""
    
    def __init__(self, n_osc: int = 8):
        super().__init__(
            n_osc=n_osc,
            bandwidth=50.0,      # Narrower for voice
            dt=0.05,            # Faster updates
            noise_strength=0.02  # More noise for natural speech
        )
        self.K_min = 0.2    # Higher minimum coupling
        self.K_max = 0.8    # Lower maximum (speech is less periodic)

class MusicOscillator(BanksyOscillator):
    """Oscillator tuned for music processing"""
    
    def __init__(self, n_osc: int = 16):
        super().__init__(
            n_osc=n_osc,
            bandwidth=200.0,     # Wider for harmonics
            dt=0.02,            # Very fast updates
            noise_strength=0.005 # Less noise for clean tones
        )
        self.K_min = 0.3    # Strong minimum coupling
        self.K_max = 1.5    # Very strong for pure tones

# Utility functions

def create_oscillator_for_context(context: str = "general") -> BanksyOscillator:
    """Factory function to create appropriate oscillator type"""
    if context == "voice":
        return VoiceOscillator()
    elif context == "music":
        return MusicOscillator()
    else:
        return BanksyOscillator()

def visualize_phases(phases: np.ndarray) -> str:
    """ASCII visualization of phase distribution"""
    n = len(phases)
    vis = ['.' for _ in range(12)]
    
    for phase in phases:
        idx = int((phase / (2 * np.pi)) * 12) % 12
        vis[idx] = 'o'
    
    return ''.join(vis)
