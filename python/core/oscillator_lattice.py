"""
Enhanced Oscillator Lattice for TORI System - BULLETPROOF EDITION
Provides wave synchronization and phase coupling with vectorized computation
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Callable, Union
from datetime import datetime
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

logger = logging.getLogger(__name__)


class OscillatorLattice:
    """
    Enhanced Oscillator Lattice for TORI's cognitive resonance system
    Manages synchronized oscillations and phase coupling
    
    Features:
    - Vectorized computation for high performance
    - Adaptive time stepping with monotonic clock
    - Support for custom adjacency matrices
    - Thread-safe operations
    - Multiple integration methods
    """
    
    def __init__(self, 
                 size: int = 64, 
                 coupling_strength: float = 0.1,
                 dt: float = 0.01,
                 adjacency: Optional[np.ndarray] = None,
                 integrator: str = "euler"):
        """
        Initialize the oscillator lattice
        
        Args:
            size: Number of oscillators in the lattice
            coupling_strength: Strength of coupling between oscillators
            dt: Time step for integration
            adjacency: Optional adjacency matrix for custom coupling topology
            integrator: Integration method ("euler" or "rk4")
        """
        self.size = size
        self.coupling_strength = coupling_strength
        self.oscillators = np.random.random(size) * 2 * np.pi  # Phase angles
        self.frequencies = np.ones(size) + np.random.random(size) * 0.1  # Natural frequencies
        self.amplitudes = np.ones(size)
        self.running = False  # Start as not running
        self.dt = dt  # Configurable time step
        self.lock = threading.Lock()
        
        # Custom adjacency matrix for weighted coupling
        if adjacency is not None and adjacency.shape == (size, size):
            self.adjacency = adjacency
        else:
            # Default to all-to-all coupling
            self.adjacency = None
        
        # Integrator selection
        if integrator not in ["euler", "rk4"]:
            logger.warning(f"Unknown integrator '{integrator}', falling back to 'euler'")
            integrator = "euler"
        self.integrator = integrator
        
        # Thread management
        self.thread = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # Event callbacks
        self.step_callbacks = []
        
        logger.info(f"✅ OscillatorLattice initialized with {size} oscillators (integrator={integrator})")
        logger.info("🌊 Oscillator lattice using centralized BPS configuration")
    
    def start(self):
        """Start the oscillator lattice"""
        with self.lock:
            if not self.running:
                self.running = True
                # Use executor for better thread management
                self.future = self.executor.submit(self._run_loop)
                logger.info("🌊 Oscillator lattice started")
    
    def stop(self):
        """Stop the oscillator lattice and clean up threads"""
        with self.lock:
            if self.running:
                self.running = False
                logger.info("⏹️ Oscillator lattice stopped")
                
                # Wait for thread to complete
                if self.future and not self.future.done():
                    try:
                        # Give the thread a chance to exit gracefully
                        self.future.result(timeout=1.0)
                    except:
                        pass
    
    def _run_loop(self):
        """
        Main oscillator evolution loop with precise timing
        Uses monotonic clock for stable real-time pacing
        """
        next_t = time.perf_counter()
        iterations = 0
        
        while self.running:
            # Evolve the system
            self._evolve_step()
            
            # Calculate next wake time
            next_t += self.dt
            
            # Notify callbacks with the current state
            if self.step_callbacks and iterations % 5 == 0:  # Notify every 5 steps
                try:
                    state = self.get_state()
                    for callback in self.step_callbacks:
                        callback(state)
                except Exception as e:
                    logger.error(f"Error in step callback: {e}")
            
            # Sleep precisely until next step time
            sleep_time = max(0, next_t - time.perf_counter())
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -0.05:  # We're falling behind by more than 50ms
                logger.debug(f"Oscillator lattice falling behind: {-sleep_time:.1f}ms")
                # Reset timing if we fall too far behind
                next_t = time.perf_counter() + self.dt
            
            iterations += 1
    
    def _evolve_step(self):
        """
        Vectorized implementation of oscillator evolution
        Uses NumPy broadcasting for massive speedup
        """
        with self.lock:
            if self.integrator == "rk4":
                # RK4 integration for better accuracy with higher coupling strengths
                self._rk4_step()
            else:
                # Default Euler integration
                self._euler_step()
    
    def _euler_step(self):
        """Euler integration step"""
        # Calculate all pairwise phase differences using broadcasting
        delta = self.oscillators[:, None] - self.oscillators
        
        # Calculate coupling terms (sin of phase differences)
        coupling_term = np.sin(-delta)  # Sign matches original implementation
        
        # Apply adjacency mask if provided
        if self.adjacency is not None:
            coupling_term *= self.adjacency
        
        # Sum coupling terms for each oscillator
        coupling_sum = coupling_term.sum(axis=1)
        
        # Update phases
        self.oscillators += self.dt * (
            self.frequencies + 
            (self.coupling_strength / self.size) * coupling_sum
        )
        
        # Keep phases in [0, 2π]
        self.oscillators %= 2 * np.pi
        
        # Evolve amplitudes if needed (Stuart-Landau dynamics)
        # Commented out for now, but can be enabled for amplitude dynamics
        # self.amplitudes += self.dt * (
        #     (1.0 - self.amplitudes**2) * self.amplitudes
        # )
    
    def _rk4_step(self):
        """
        4th-order Runge-Kutta integration for better stability
        with high coupling strengths or complex dynamics
        """
        def derivatives(phases):
            """Calculate derivatives for the system"""
            delta = phases[:, None] - phases
            coupling_term = np.sin(-delta)
            
            if self.adjacency is not None:
                coupling_term *= self.adjacency
                
            coupling_sum = coupling_term.sum(axis=1)
            
            return self.frequencies + (self.coupling_strength / self.size) * coupling_sum
        
        # RK4 integration steps
        y0 = self.oscillators
        
        k1 = derivatives(y0)
        k2 = derivatives(y0 + 0.5 * self.dt * k1)
        k3 = derivatives(y0 + 0.5 * self.dt * k2)
        k4 = derivatives(y0 + self.dt * k3)
        
        # Update phases
        self.oscillators = (y0 + (self.dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)) % (2 * np.pi)
    
    def get_state(self) -> Dict:
        """Get current oscillator state"""
        with self.lock:
            # Calculate metrics for monitoring
            phases = self.oscillators
            sync_order = self._calculate_synchronization()
            
            return {
                'phases': phases.tolist(),
                'amplitudes': self.amplitudes.tolist(),
                'frequencies': self.frequencies.tolist(),
                'coupling_strength': self.coupling_strength,
                'running': self.running,
                'synchronization': sync_order,
                'available': True,
                'status': 'active',
                # Monitoring metrics
                'metrics': {
                    'synchronization': sync_order,
                    'mean_frequency': float(np.mean(self.frequencies)),
                    'std_phase': float(np.std(phases)),
                    'integrator': self.integrator,
                    'dt': self.dt
                }
            }
    
    def _calculate_synchronization(self) -> float:
        """Calculate order parameter (synchronization measure)"""
        z = np.mean(np.exp(1j * self.oscillators))
        return float(abs(z))
    
    def set_external_drive(self, oscillator_idx: int, frequency: float):
        """Set external driving frequency for specific oscillator"""
        if 0 <= oscillator_idx < self.size:
            with self.lock:
                self.frequencies[oscillator_idx] = frequency
                logger.debug(f"Set oscillator {oscillator_idx} frequency to {frequency}")
    
    def set_frequencies(self, frequencies: np.ndarray):
        """Set frequencies for all oscillators at once"""
        if len(frequencies) != self.size:
            raise ValueError(f"Expected {self.size} frequencies, got {len(frequencies)}")
            
        with self.lock:
            self.frequencies = np.array(frequencies, dtype=float)
            logger.debug(f"Set all oscillator frequencies")
    
    def set_amplitudes(self, amplitudes: np.ndarray):
        """Set amplitudes for all oscillators at once"""
        if len(amplitudes) != self.size:
            raise ValueError(f"Expected {self.size} amplitudes, got {len(amplitudes)}")
            
        with self.lock:
            self.amplitudes = np.array(amplitudes, dtype=float)
            logger.debug(f"Set all oscillator amplitudes")
    
    def inject_perturbation(self, phase_shift: float):
        """Inject phase perturbation to all oscillators"""
        with self.lock:
            self.oscillators += phase_shift
            self.oscillators %= 2 * np.pi
            logger.debug(f"Injected phase perturbation: {phase_shift}")
    
    def inject_targeted_perturbation(self, indices: List[int], phase_shifts: Union[float, List[float]]):
        """Inject phase perturbation to specific oscillators"""
        with self.lock:
            if isinstance(phase_shifts, (int, float)):
                # Same shift for all targeted oscillators
                for idx in indices:
                    if 0 <= idx < self.size:
                        self.oscillators[idx] += phase_shifts
            else:
                # Individual shifts
                for idx, shift in zip(indices, phase_shifts):
                    if 0 <= idx < self.size and idx < len(phase_shifts):
                        self.oscillators[idx] += shift
                        
            self.oscillators %= 2 * np.pi
            logger.debug(f"Injected targeted perturbation to {len(indices)} oscillators")
    
    def get_hologram_data(self) -> Dict:
        """Get data formatted for hologram visualization"""
        state = self.get_state()
        
        # Convert to visualization format
        x = np.cos(state['phases']) * state['amplitudes']
        y = np.sin(state['phases']) * state['amplitudes']
        z = np.array(state['frequencies']) - 1.0  # Center around 0
        
        return {
            'positions': {
                'x': x.tolist(),
                'y': y.tolist(), 
                'z': z.tolist()
            },
            'phases': state['phases'],
            'synchronization': state['synchronization'],
            'timestamp': datetime.now().isoformat()
        }
    
    def reset(self):
        """Reset oscillators to random initial conditions"""
        with self.lock:
            self.oscillators = np.random.random(self.size) * 2 * np.pi
            self.frequencies = np.ones(self.size) + np.random.random(self.size) * 0.1
            logger.info("🔄 Oscillator lattice reset to random initial conditions")
    
    def register_step_callback(self, callback: Callable[[Dict], None]):
        """Register callback for step events"""
        self.step_callbacks.append(callback)
        logger.debug(f"Registered step callback, total callbacks: {len(self.step_callbacks)}")
    
    def remove_step_callback(self, callback: Callable[[Dict], None]):
        """Remove callback from step events"""
        if callback in self.step_callbacks:
            self.step_callbacks.remove(callback)
            logger.debug(f"Removed step callback, remaining callbacks: {len(self.step_callbacks)}")


# Global lattice instance - initialized on first use with lock for thread safety
_global_lattice = None
_global_lattice_lock = threading.Lock()


def get_global_lattice():
    """
    Get the global oscillator lattice instance
    Thread-safe accessor with proper error handling
    """
    global _global_lattice, _global_lattice_lock
    
    with _global_lattice_lock:
        if _global_lattice is None:
            try:
                _global_lattice = OscillatorLattice()
                _global_lattice.start()
                logger.info("🌊 Global oscillator lattice initialized and started")
            except Exception as e:
                logger.error(f"Failed to create global lattice: {e}")
                # Re-raise the exception rather than silently returning None
                raise
    
    return _global_lattice


def get_oscillator_lattice(size: int = 64) -> OscillatorLattice:
    """Get or create global oscillator lattice instance"""
    return get_global_lattice()  # Use same function to ensure single instance


def initialize_global_lattice(config=None) -> bool:
    """
    Initialize the global oscillator lattice with configuration
    Thread-safe and idempotent
    """
    global _global_lattice, _global_lattice_lock
    
    with _global_lattice_lock:
        if _global_lattice is not None:
            logger.info("Global oscillator lattice already initialized")
            return True
        
        try:
            # Extract configuration options
            config = config or {}
            size = config.get('size', 64)
            coupling_strength = config.get('coupling_strength', 0.1)
            dt = config.get('dt', 0.01)
            integrator = config.get('integrator', 'euler')
            
            # Load adjacency matrix if specified
            adjacency = None
            adjacency_file = config.get('adjacency_file')
            if adjacency_file:
                try:
                    adjacency = np.load(adjacency_file)
                    logger.info(f"Loaded adjacency matrix from {adjacency_file}")
                except Exception as e:
                    logger.error(f"Failed to load adjacency matrix: {e}")
            
            # Create and start the lattice
            _global_lattice = OscillatorLattice(
                size=size,
                coupling_strength=coupling_strength,
                dt=dt,
                adjacency=adjacency,
                integrator=integrator
            )
            _global_lattice.start()
            
            logger.info(f"🌊 Global oscillator lattice initialized and started (size={size}, dt={dt})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize global lattice: {e}")
            return False


def shutdown_oscillator_lattice():
    """
    Shutdown global oscillator lattice cleanly
    Ensures thread is properly joined
    """
    global _global_lattice, _global_lattice_lock
    
    with _global_lattice_lock:
        if _global_lattice is not None:
            _global_lattice.stop()
            _global_lattice = None
            logger.info("🛑 Global oscillator lattice shutdown")


# Export public API
__all__ = [
    'OscillatorLattice',
    'get_global_lattice',
    'get_oscillator_lattice', 
    'initialize_global_lattice',
    'shutdown_oscillator_lattice'
]

# Register shutdown hook for FastAPI integration
try:
    from fastapi import FastAPI
    
    def register_lattice_shutdown_hook(app: FastAPI):
        """Register shutdown hook for FastAPI"""
        @app.on_event("shutdown")
        async def shutdown_hook():
            logger.info("FastAPI shutting down, cleaning up oscillator lattice")
            shutdown_oscillator_lattice()
            
        logger.info("Registered oscillator lattice shutdown hook with FastAPI")
    
    __all__.append('register_lattice_shutdown_hook')
except ImportError:
    # FastAPI not available
    pass
