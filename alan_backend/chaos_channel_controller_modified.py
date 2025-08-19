#!/usr/bin/env python3
"""
Modified chaos_channel_controller.py - Enhanced with observer token emission
Production-ready implementation with comprehensive integration
"""

import asyncio
import numpy as np
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import logging
from enum import Enum
from collections import deque
import json

# NEW IMPORTS for No-DB integration
from python.core.torus_registry import get_torus_registry
from python.core.observer_synthesis import emit_token

logger = logging.getLogger(__name__)

# Chaos burst parameters
MAX_BURST_DURATION = 500  # Maximum burst duration in steps
MAX_BURST_LEVEL = 1.0     # Maximum chaos level
COOLDOWN_PERIOD = 100     # Steps between bursts
ENERGY_DECAY_RATE = 0.05  # Energy decay per step

class BurstState(Enum):
    """States of chaos burst"""
    IDLE = "idle"
    ACTIVE = "active"
    COOLDOWN = "cooldown"

@dataclass
class ChaosBurst:
    """Represents a chaos burst event"""
    burst_id: str
    level: float
    duration: int
    start_time: datetime
    purpose: str = "exploration"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class BurstMetrics:
    """Metrics for a completed burst"""
    burst_id: str
    actual_duration: int
    peak_energy: float
    energy_returned_ratio: float
    discoveries: List[Dict[str, Any]] = field(default_factory=list)
    tokens_emitted: int = 0

class ChaosChannelController:
    """
    Controller for managed chaos bursts
    Provides safe, time-limited chaos exploration with observer token emission
    """
    
    def __init__(self, lattice_size: int = 128):
        self.lattice_size = lattice_size
        self.state = BurstState.IDLE
        
        # Current burst tracking
        self.current_burst: Optional[ChaosBurst] = None
        self.burst_step = 0
        self.cooldown_remaining = 0
        
        # Energy management
        self.baseline_energy = 1.0
        self.current_energy = 1.0
        self.energy_history = deque(maxlen=1000)
        
        # Burst history
        self.burst_history: List[BurstMetrics] = []
        self.active_callbacks: List[Callable] = []
        
        # Lattice state
        self.lattice_state = np.ones((lattice_size, lattice_size), dtype=complex)
        
        # Torus registry for persistence
        self.torus_registry = get_torus_registry()
        
        # Metrics
        self.metrics = {
            'bursts_triggered': 0,
            'total_discoveries': 0,
            'tokens_emitted': 0,
            'registry_records': 0
        }
        
        logger.info(f"ChaosChannelController initialized with {lattice_size}x{lattice_size} lattice")
        
    def trigger(self, level: float, duration: int, 
                purpose: str = "exploration",
                callback: Optional[Callable] = None) -> str:
        """
        Trigger a chaos burst with observer token emission
        
        Args:
            level: Chaos intensity (0.0 to 1.0)
            duration: Burst duration in steps
            purpose: Purpose of burst (for logging)
            callback: Optional callback when burst completes
            
        Returns:
            burst_id if triggered, empty string if rejected
        """
        # Validate parameters
        level = np.clip(level, 0.0, MAX_BURST_LEVEL)
        duration = min(duration, MAX_BURST_DURATION)
        
        # Check if available
        if self.state != BurstState.IDLE:
            logger.warning(f"Cannot trigger burst - current state: {self.state}")
            return ""
            
        # Create burst
        burst_id = f"burst_{datetime.now(timezone.utc).timestamp():.6f}"
        self.current_burst = ChaosBurst(
            burst_id=burst_id,
            level=level,
            duration=duration,
            start_time=datetime.now(timezone.utc),
            purpose=purpose
        )
        
        # Update state
        self.state = BurstState.ACTIVE
        self.burst_step = 0
        self.metrics['bursts_triggered'] += 1
        
        # Register callback
        if callback:
            self.active_callbacks.append(callback)
            
        # Initialize burst energy
        self._inject_chaos_energy(level)
        
        # Emit burst start token
        try:
            token = emit_token({
                "type": "chaos_burst_start",
                "source": "chaos_controller",
                "burst_id": burst_id,
                "level": float(level),
                "duration": duration,
                "purpose": purpose,
                "baseline_energy": float(self.baseline_energy),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            self.metrics['tokens_emitted'] += 1
            logger.debug(f"Emitted burst start token: {token[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to emit burst start token: {e}")
        
        logger.info(f"Triggered chaos burst {burst_id}: level={level}, duration={duration}, purpose={purpose}")
        
        return burst_id
        
    def step(self) -> Dict[str, Any]:
        """
        Advance chaos burst by one step
        Returns current metrics
        """
        metrics = {
            'state': self.state.value,
            'energy': float(self.current_energy),
            'burst_active': self.state == BurstState.ACTIVE
        }
        
        if self.state == BurstState.ACTIVE:
            # Active burst
            self.burst_step += 1
            
            # Evolve chaotic dynamics
            self._evolve_chaos()
            
            # Check if burst complete
            if self.burst_step >= self.current_burst.duration:
                self._complete_burst()
                
            metrics['burst_progress'] = self.burst_step / self.current_burst.duration
            metrics['burst_id'] = self.current_burst.burst_id
            
        elif self.state == BurstState.COOLDOWN:
            # Cooldown period
            self.cooldown_remaining -= 1
            
            # Decay energy toward baseline
            self.current_energy = self.baseline_energy + \
                (self.current_energy - self.baseline_energy) * (1 - ENERGY_DECAY_RATE)
                
            if self.cooldown_remaining <= 0:
                self.state = BurstState.IDLE
                logger.info("Cooldown complete - ready for next burst")
                
                # Emit cooldown complete token
                try:
                    token = emit_token({
                        "type": "chaos_cooldown_complete",
                        "source": "chaos_controller",
                        "energy": float(self.current_energy),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    self.metrics['tokens_emitted'] += 1
                except Exception as e:
                    logger.warning(f"Failed to emit cooldown token: {e}")
                
            metrics['cooldown_remaining'] = self.cooldown_remaining
            
        else:
            # Idle - maintain baseline
            self.current_energy = self.baseline_energy
            
        # Record energy
        self.energy_history.append(self.current_energy)
        
        return metrics
        
    def _inject_chaos_energy(self, level: float):
        """Inject chaos energy into the lattice"""
        # Create chaotic perturbation
        noise = np.random.randn(self.lattice_size, self.lattice_size) + \
                1j * np.random.randn(self.lattice_size, self.lattice_size)
                
        # Scale by level
        perturbation = noise * level * 0.1
        
        # Add to lattice with spatial modulation
        x = np.linspace(-1, 1, self.lattice_size)
        y = np.linspace(-1, 1, self.lattice_size)
        X, Y = np.meshgrid(x, y)
        
        # Gaussian envelope to concentrate chaos
        envelope = np.exp(-(X**2 + Y**2) / 0.5)
        
        self.lattice_state += perturbation * envelope
        
        # Update energy
        self.current_energy = self.baseline_energy * (1 + level)
        
        # Record initial state in registry
        try:
            shape_id = self.torus_registry.record_shape(
                vertices=self.lattice_state.flatten()[:100],  # Sample for storage
                coherence_band='critical',  # Chaos injection is critical
                metadata={
                    'type': 'chaos_injection',
                    'burst_id': self.current_burst.burst_id if self.current_burst else None,
                    'level': float(level),
                    'energy': float(self.current_energy)
                }
            )
            self.metrics['registry_records'] += 1
            logger.debug(f"Recorded chaos injection state: {shape_id}")
        except Exception as e:
            logger.warning(f"Failed to record chaos injection: {e}")
        
    def _evolve_chaos(self):
        """Evolve chaotic dynamics during burst"""
        # Nonlinear evolution with coupling
        dt = 0.01
        
        # Laplacian for diffusion
        laplacian = np.roll(self.lattice_state, 1, axis=0) + \
                   np.roll(self.lattice_state, -1, axis=0) + \
                   np.roll(self.lattice_state, 1, axis=1) + \
                   np.roll(self.lattice_state, -1, axis=1) - \
                   4 * self.lattice_state
                   
        # Nonlinear term
        nonlinear = self.lattice_state * np.abs(self.lattice_state)**2
        
        # Evolution equation
        self.lattice_state += dt * (0.5 * laplacian - nonlinear)
        
        # Add small noise to maintain chaos
        if self.current_burst:
            noise_level = self.current_burst.level * 0.01
            self.lattice_state += noise_level * (
                np.random.randn(self.lattice_size, self.lattice_size) +
                1j * np.random.randn(self.lattice_size, self.lattice_size)
            )
            
        # Update energy
        self.current_energy = float(np.mean(np.abs(self.lattice_state)**2))
        
        # Emit evolution token periodically
        if self.burst_step % 50 == 0:  # Every 50 steps
            try:
                token = emit_token({
                    "type": "chaos_evolution",
                    "source": "chaos_controller",
                    "burst_id": self.current_burst.burst_id,
                    "step": self.burst_step,
                    "energy": float(self.current_energy),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                self.metrics['tokens_emitted'] += 1
            except Exception as e:
                logger.warning(f"Failed to emit evolution token: {e}")
        
    def _complete_burst(self):
        """Complete the current burst with comprehensive metrics"""
        if not self.current_burst:
            return
            
        # Calculate metrics
        peak_energy = float(max(self.energy_history)) if self.energy_history else self.baseline_energy
        final_energy = self.current_energy
        
        # Look for discoveries
        discoveries = self._detect_discoveries()
        
        metrics = BurstMetrics(
            burst_id=self.current_burst.burst_id,
            actual_duration=self.burst_step,
            peak_energy=peak_energy,
            energy_returned_ratio=float(final_energy / self.baseline_energy),
            discoveries=discoveries,
            tokens_emitted=0  # Will be updated
        )
        
        # Update discovery count
        self.metrics['total_discoveries'] += len(discoveries)
        
        # Record final state in registry
        try:
            shape_id = self.torus_registry.record_shape(
                vertices=self.lattice_state.flatten()[:100],  # Sample
                coherence_band='local' if final_energy < 1.5 else 'global',
                metadata={
                    'type': 'chaos_burst_complete',
                    'burst_id': self.current_burst.burst_id,
                    'discoveries': len(discoveries),
                    'peak_energy': peak_energy,
                    'energy_ratio': metrics.energy_returned_ratio
                }
            )
            self.metrics['registry_records'] += 1
            logger.debug(f"Recorded burst completion state: {shape_id}")
        except Exception as e:
            logger.warning(f"Failed to record burst completion: {e}")
        
        # Emit observer token for burst completion
        try:
            token = emit_token({
                "type": "chaos_burst_complete",
                "source": "chaos_controller",
                "burst_id": self.current_burst.burst_id,
                "peak_energy": peak_energy,
                "energy_returned_ratio": metrics.energy_returned_ratio,
                "discoveries": len(discoveries),
                "discovery_types": [d['type'] for d in discoveries],
                "duration": self.burst_step,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            metrics.tokens_emitted = 1
            self.metrics['tokens_emitted'] += 1
            logger.debug(f"Emitted burst completion token: {token[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to emit burst token: {e}")
        
        # Store metrics
        self.burst_history.append(metrics)
        
        # Notify callbacks
        for callback in self.active_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"Callback error: {e}")
                
        self.active_callbacks.clear()
        
        # Enter cooldown
        self.state = BurstState.COOLDOWN
        self.cooldown_remaining = COOLDOWN_PERIOD
        self.current_burst = None
        
        logger.info(f"Burst complete: {metrics.burst_id}, peak_energy={peak_energy:.3f}, discoveries={len(discoveries)}")
        
    def _detect_discoveries(self) -> List[Dict[str, Any]]:
        """Detect interesting patterns or structures"""
        discoveries = []
        
        # Check for soliton-like structures
        amplitude = np.abs(self.lattice_state)
        
        # Find local maxima
        try:
            from scipy.ndimage import maximum_filter
            local_max = (amplitude == maximum_filter(amplitude, size=5))
            
            # Count significant peaks
            threshold = 2 * self.baseline_energy
            peaks = np.where(local_max & (amplitude > threshold))
            
            if len(peaks[0]) > 0:
                discoveries.append({
                    'type': 'soliton_candidates',
                    'count': len(peaks[0]),
                    'locations': [(int(x), int(y)) for x, y in zip(peaks[0][:5], peaks[1][:5])],
                    'max_amplitude': float(np.max(amplitude)),
                    'mean_amplitude': float(np.mean(amplitude[peaks]))
                })
        except ImportError:
            # scipy not available - use simple detection
            max_amp = float(np.max(amplitude))
            if max_amp > 2 * self.baseline_energy:
                discoveries.append({
                    'type': 'high_amplitude_regions',
                    'max_amplitude': max_amp
                })
            
        # Check for phase vortices
        phase = np.angle(self.lattice_state)
        phase_grad_x = np.gradient(phase, axis=1)
        phase_grad_y = np.gradient(phase, axis=0)
        
        # Circulation (simplified vortex detection)
        circulation = np.abs(phase_grad_x[1:, :-1] - phase_grad_x[:-1, :-1] +
                           phase_grad_y[:-1, 1:] - phase_grad_y[:-1, :-1])
                           
        vortex_candidates = int(np.sum(circulation > np.pi))
        
        if vortex_candidates > 0:
            discoveries.append({
                'type': 'phase_vortices',
                'count': vortex_candidates,
                'total_circulation': float(np.sum(circulation)),
                'mean_circulation': float(np.mean(circulation[circulation > np.pi]))
            })
            
        # Check for coherent structures
        coherence = np.abs(np.fft.fft2(self.lattice_state))
        coherence_peak = float(np.max(coherence))
        
        if coherence_peak > 10 * self.baseline_energy:
            discoveries.append({
                'type': 'coherent_structure',
                'coherence_peak': coherence_peak,
                'spectral_concentration': float(coherence_peak / np.sum(coherence))
            })
            
        return discoveries
        
    def get_lattice_state(self) -> np.ndarray:
        """Get current lattice state"""
        return self.lattice_state.copy()
        
    def get_energy_history(self, n_recent: int = 100) -> List[float]:
        """Get recent energy history"""
        return [float(e) for e in list(self.energy_history)[-n_recent:]]
        
    def get_burst_history(self) -> List[BurstMetrics]:
        """Get history of completed bursts"""
        return self.burst_history.copy()
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            'metrics': self.metrics.copy(),
            'total_bursts': len(self.burst_history),
            'current_state': self.state.value,
            'energy': {
                'current': float(self.current_energy),
                'baseline': float(self.baseline_energy),
                'history_size': len(self.energy_history)
            },
            'discoveries': {
                'total': self.metrics['total_discoveries'],
                'by_type': self._count_discoveries_by_type()
            },
            'registry_records': self.metrics['registry_records']
        }
        
    def _count_discoveries_by_type(self) -> Dict[str, int]:
        """Count discoveries by type across all bursts"""
        counts = {}
        for burst in self.burst_history:
            for discovery in burst.discoveries:
                d_type = discovery['type']
                counts[d_type] = counts.get(d_type, 0) + 1
        return counts
        
    def reset(self):
        """Reset controller to initial state"""
        self.state = BurstState.IDLE
        self.current_burst = None
        self.burst_step = 0
        self.cooldown_remaining = 0
        self.current_energy = self.baseline_energy
        self.lattice_state = np.ones((self.lattice_size, self.lattice_size), dtype=complex)
        self.energy_history.clear()
        self.burst_history.clear()
        self.active_callbacks.clear()
        
        # Don't reset metrics - keep historical counts
        
        logger.info("ChaosChannelController reset to initial state")
        
# Global controller instance
_controller = None

def get_controller(lattice_size: int = 128) -> ChaosChannelController:
    """Get or create the global controller instance"""
    global _controller
    if _controller is None:
        _controller = ChaosChannelController(lattice_size)
    return _controller

# Unit test
def test_chaos_burst():
    """Test chaos burst functionality with observer tokens"""
    print("Testing chaos burst controller with observer tokens...")
    
    controller = ChaosChannelController(lattice_size=64)
    
    # Record initial metrics
    initial_tokens = controller.metrics['tokens_emitted']
    initial_energy = controller.current_energy
    
    # Trigger burst
    burst_id = controller.trigger(level=0.3, duration=50, purpose="test")
    assert burst_id != "", "Burst should be triggered"
    print(f"Triggered burst: {burst_id}")
    
    # Run burst
    peak_energy = initial_energy
    for i in range(100):  # Run past burst duration
        metrics = controller.step()
        if metrics['energy'] > peak_energy:
            peak_energy = metrics['energy']
            
    # Check we're in cooldown
    assert controller.state == BurstState.COOLDOWN, "Should be in cooldown"
    
    # Check energy returned close to baseline
    final_metrics = controller.step()
    energy_ratio = final_metrics['energy'] / initial_energy
    
    print(f"\nResults:")
    print(f"Initial energy: {initial_energy:.3f}")
    print(f"Peak energy: {peak_energy:.3f}")
    print(f"Final energy: {final_metrics['energy']:.3f}")
    print(f"Energy ratio: {energy_ratio:.3f}")
    print(f"Tokens emitted: {controller.metrics['tokens_emitted'] - initial_tokens}")
    print(f"Registry records: {controller.metrics['registry_records']}")
    
    # Get statistics
    stats = controller.get_statistics()
    print(f"\nStatistics:")
    print(f"Total discoveries: {stats['discoveries']['total']}")
    print(f"Discovery types: {stats['discoveries']['by_type']}")
    
    # Check energy conservation (within 5%)
    assert 0.95 <= energy_ratio <= 1.05, f"Energy ratio {energy_ratio} out of bounds"
    
    print("\n✓ Chaos burst test PASSED")
    
    return True

if __name__ == "__main__":
    # Ensure state directory exists
    import os
    if not os.getenv('TORI_STATE_ROOT'):
        os.environ['TORI_STATE_ROOT'] = './tori_state_test'
    
    # Run test
    test_chaos_burst()
    
    # Demo with visualization
    try:
        import matplotlib.pyplot as plt
        
        controller = ChaosChannelController()
        
        # Trigger burst
        burst_id = controller.trigger(level=0.5, duration=100, purpose="demo")
        
        # Collect data
        energy_history = []
        
        for step in range(200):
            metrics = controller.step()
            energy_history.append(metrics['energy'])
            
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(energy_history)
        plt.axhline(y=controller.baseline_energy, color='r', linestyle='--', label='Baseline')
        plt.axvline(x=100, color='g', linestyle='--', label='Burst end')
        plt.xlabel('Step')
        plt.ylabel('Energy')
        plt.title('Chaos Burst Energy Profile with Observer Tokens')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('chaos_burst_profile_with_tokens.png')
        print("\nSaved energy profile to chaos_burst_profile_with_tokens.png")
        
        # Plot phase space
        state = controller.get_lattice_state()
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(np.abs(state), cmap='hot')
        plt.colorbar()
        plt.title('Amplitude')
        
        plt.subplot(1, 2, 2)
        plt.imshow(np.angle(state), cmap='hsv')
        plt.colorbar()
        plt.title('Phase')
        
        plt.tight_layout()
        plt.savefig('chaos_burst_final_state.png')
        print("Saved final state to chaos_burst_final_state.png")
        
    except ImportError:
        print("\nMatplotlib not available - skipping visualization")
    
    print("\n✅ All tests completed!")
