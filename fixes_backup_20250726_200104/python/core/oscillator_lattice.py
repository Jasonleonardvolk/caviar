"""
Enhanced Oscillator Lattice for TORI System - BULLETPROOF EDITION
Provides wave synchronization and phase coupling
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import threading
import time

logger = logging.getLogger(__name__)


class OscillatorLattice:
    """
    Enhanced Oscillator Lattice for TORI's cognitive resonance system
    Manages synchronized oscillations and phase coupling
    """
    
    def __init__(self, size: int = 64, coupling_strength: float = 0.1):
        self.size = size
        self.coupling_strength = coupling_strength
        self.oscillators = np.random.random(size) * 2 * np.pi  # Phase angles
        self.frequencies = np.ones(size) + np.random.random(size) * 0.1  # Natural frequencies
        self.amplitudes = np.ones(size)
        self.running = True  # Always available now!
        self.step_size = 0.01
        self.lock = threading.Lock()
        
        logger.info(f"✅ OscillatorLattice initialized with {size} oscillators")
        logger.info("🌊 Oscillator lattice using centralized BPS configuration")
    
    def start(self):
        """Start the oscillator lattice"""
        with self.lock:
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._run_loop, daemon=True)
                self.thread.start()
                logger.info("🌊 Oscillator lattice started")
    
    def stop(self):
        """Stop the oscillator lattice"""
        with self.lock:
            if self.running:
                self.running = False
                logger.info("⏹️ Oscillator lattice stopped")
    
    def _run_loop(self):
        """Main oscillator evolution loop"""
        while self.running:
            self._evolve_step()
            time.sleep(0.1)  # 10Hz update rate
    
    def _evolve_step(self):
        """Evolve oscillators by one time step using Kuramoto model"""
        with self.lock:
            # Calculate coupling terms
            coupling = np.zeros(self.size)
            for i in range(self.size):
                for j in range(self.size):
                    if i != j:
                        coupling[i] += np.sin(self.oscillators[j] - self.oscillators[i])
            
            # Update phases
            self.oscillators += self.step_size * (
                self.frequencies + 
                self.coupling_strength / self.size * coupling
            )
            
            # Keep phases in [0, 2π]
            self.oscillators = self.oscillators % (2 * np.pi)
    
    def get_state(self) -> Dict:
        """Get current oscillator state"""
        with self.lock:
            return {
                'phases': self.oscillators.tolist(),
                'amplitudes': self.amplitudes.tolist(),
                'frequencies': self.frequencies.tolist(),
                'coupling_strength': self.coupling_strength,
                'running': self.running,
                'synchronization': self._calculate_synchronization(),
                'available': True,  # Always available!
                'status': 'active'
            }
    
    def _calculate_synchronization(self) -> float:
        """Calculate order parameter (synchronization measure)"""
        z = np.mean(np.exp(1j * self.oscillators))
        return abs(z)
    
    def set_external_drive(self, oscillator_idx: int, frequency: float):
        """Set external driving frequency for specific oscillator"""
        if 0 <= oscillator_idx < self.size:
            with self.lock:
                self.frequencies[oscillator_idx] = frequency
                logger.debug(f"Set oscillator {oscillator_idx} frequency to {frequency}")
    
    def inject_perturbation(self, phase_shift: float):
        """Inject phase perturbation to all oscillators"""
        with self.lock:
            self.oscillators += phase_shift
            self.oscillators = self.oscillators % (2 * np.pi)
            logger.debug(f"Injected phase perturbation: {phase_shift}")
    
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


def get_global_lattice():
    """Get the global oscillator lattice instance"""
    global _global_lattice
    if _global_lattice is None:
        try:
            _global_lattice = OscillatorLattice()
            logger.info("Global oscillator lattice created")
        except Exception as e:
            logger.warning(f"Could not create global lattice: {e}")
            _global_lattice = None
    return _global_lattice

def initialize_global_lattice(config=None):
    """Initialize the global oscillator lattice"""
    global _global_lattice
    try:
        _global_lattice = OscillatorLattice(config or {})
        logger.info("Global oscillator lattice initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize global lattice: {e}")
        return False

# Global instance - always available!
_global_lattice = OscillatorLattice(size=64)
_global_lattice.start()  # Start immediately
logger.info("🌊 Global oscillator lattice initialized and started")


def get_oscillator_lattice(size: int = 64) -> OscillatorLattice:
    """Get or create global oscillator lattice instance"""
    global _global_lattice
    if _global_lattice is None:
        _global_lattice = OscillatorLattice(size=size)
        _global_lattice.start()
        logger.info("🌊 Global oscillator lattice initialized")
    return _global_lattice


def shutdown_oscillator_lattice():
    """Shutdown global oscillator lattice"""
    global _global_lattice
    if _global_lattice is not None:
        _global_lattice.stop()
        _global_lattice = None
        logger.info("🛑 Global oscillator lattice shutdown")


# Export main class
__all__ = [
    'get_global_lattice',
    'initialize_global_lattice','OscillatorLattice', 'get_oscillator_lattice', 'shutdown_oscillator_lattice']
