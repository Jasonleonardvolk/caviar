#!/usr/bin/env python3
"""
Chaos Channel Controller - Advanced Chaos Burst Management
Manages controlled chaos bursts with energy tracking and observer token emission
"""

import numpy as np
import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import logging

# Import torus registry for persistence
try:
    from python.core.torus_registry import TorusRegistry
    TORUS_AVAILABLE = True
except ImportError:
    TORUS_AVAILABLE = False
    logging.warning("Torus registry not available")

logger = logging.getLogger(__name__)

# Chaos burst parameters
MAX_BURST_DURATION = 5.0  # seconds
COOLDOWN_PERIOD = 30.0    # seconds between bursts
ENERGY_CONSERVATION_TOLERANCE = 0.05  # 5% tolerance
TOKEN_EMISSION_RATE = 10  # tokens per second during burst

class ChannelState(Enum):
    """States of the chaos channel"""
    IDLE = "idle"
    TRIGGERING = "triggering"
    ACTIVE = "active"
    COOLDOWN = "cooldown"
    ERROR = "error"

@dataclass
class ChaosMetrics:
    """Metrics tracked during chaos burst"""
    energy_initial: float = 0.0
    energy_current: float = 0.0
    energy_harvested: float = 0.0
    tokens_emitted: int = 0
    burst_duration: float = 0.0
    efficiency_gain: float = 0.0
    oscillations_detected: int = 0
    phase_coherence: float = 1.0

@dataclass 
class ObserverToken:
    """Token emitted during chaos observations"""
    token_id: str
    timestamp: float
    energy_snapshot: float
    phase_state: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

class ChaosChannelController:
    """
    Advanced controller for managed chaos bursts
    Provides energy tracking, token emission, and safety controls
    """
    
    def __init__(self, lattice_ref=None, observer_callback: Optional[Callable] = None):
        self.lattice_ref = lattice_ref
        self.observer_callback = observer_callback
        
        # State management
        self.state = ChannelState.IDLE
        self.last_burst_time = 0.0
        self.metrics = ChaosMetrics()
        
        # Energy tracking
        self.baseline_energy = 0.0
        self.energy_history = []
        
        # Token emission
        self.emitted_tokens: List[ObserverToken] = []
        self.token_counter = 0
        
        # Safety controls
        self.emergency_stop = False
        self.max_energy_multiplier = 10.0
        
        # Persistence
        if TORUS_AVAILABLE:
            self.registry = TorusRegistry()
        else:
            self.registry = None
            
    async def trigger(self, intensity: float = 1.0, 
                     duration: Optional[float] = None,
                     target_mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Trigger a controlled chaos burst
        
        Args:
            intensity: Burst intensity multiplier (0.1 to 5.0)
            duration: Burst duration in seconds (default: MAX_BURST_DURATION)
            target_mode: Specific chaos mode to induce
            
        Returns:
            Dict containing burst results and metrics
        """
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_burst_time < COOLDOWN_PERIOD:
            remaining = COOLDOWN_PERIOD - (current_time - self.last_burst_time)
            logger.warning(f"Cooldown active, {remaining:.1f}s remaining")
            return {'success': False, 'reason': 'cooldown', 'remaining': remaining}
            
        # Check state
        if self.state != ChannelState.IDLE:
            logger.warning(f"Cannot trigger burst in state {self.state}")
            return {'success': False, 'reason': 'invalid_state', 'current_state': self.state}
            
        # Validate parameters
        intensity = np.clip(intensity, 0.1, 5.0)
        duration = min(duration or MAX_BURST_DURATION, MAX_BURST_DURATION)
        
        # Begin burst
        self.state = ChannelState.TRIGGERING
        self.metrics = ChaosMetrics()  # Reset metrics
        
        try:
            # Capture baseline energy
            self.baseline_energy = self._calculate_system_energy()
            self.metrics.energy_initial = self.baseline_energy
            
            # Start burst
            self.state = ChannelState.ACTIVE
            burst_start = time.time()
            
            # Run chaos burst
            result = await self._execute_burst(intensity, duration, target_mode)
            
            # Calculate final metrics
            self.metrics.burst_duration = time.time() - burst_start
            self.metrics.energy_current = self._calculate_system_energy()
            self.metrics.energy_harvested = max(0, self.baseline_energy - self.metrics.energy_current)
            
            # Check energy conservation
            energy_delta = abs(self.metrics.energy_current - self.baseline_energy)
            if energy_delta / self.baseline_energy > ENERGY_CONSERVATION_TOLERANCE:
                logger.warning(f"Energy conservation violation: {energy_delta/self.baseline_energy:.2%}")
                
            # Enter cooldown
            self.state = ChannelState.COOLDOWN
            self.last_burst_time = time.time()
            
            # Schedule cooldown end
            asyncio.create_task(self._cooldown_timer())
            
            # Persist results if available
            if self.registry:
                self._persist_burst_results(result)
                
            return {
                'success': True,
                'metrics': self.metrics,
                'tokens_emitted': len(self.emitted_tokens),
                'efficiency_gain': self.metrics.efficiency_gain,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Chaos burst failed: {e}")
            self.state = ChannelState.ERROR
            return {'success': False, 'reason': 'exception', 'error': str(e)}
            
    async def _execute_burst(self, intensity: float, duration: float, 
                           target_mode: Optional[str]) -> Dict[str, Any]:
        """Execute the actual chaos burst"""
        if self.lattice_ref is None:
            # Simulation mode
            return await self._simulate_burst(intensity, duration)
            
        # Production mode with actual lattice
        burst_result = {}
        token_emission_task = asyncio.create_task(self._emit_tokens(duration))
        
        try:
            # Inject chaos into lattice
            if hasattr(self.lattice_ref, 'oscillators'):
                # Perturb oscillator phases
                for osc in self.lattice_ref.oscillators:
                    perturbation = np.random.randn() * intensity * 0.1
                    osc.phase += perturbation
                    osc.amplitude *= (1 + np.random.randn() * intensity * 0.05)
                    
            # Monitor energy evolution
            start_time = time.time()
            while time.time() - start_time < duration:
                if self.emergency_stop:
                    logger.warning("Emergency stop activated")
                    break
                    
                # Track energy
                current_energy = self._calculate_system_energy()
                self.energy_history.append(current_energy)
                self.metrics.energy_current = current_energy
                
                # Check for runaway
                if current_energy > self.baseline_energy * self.max_energy_multiplier:
                    logger.error("Energy runaway detected, aborting burst")
                    self.emergency_stop = True
                    break
                    
                # Let system evolve
                await asyncio.sleep(0.01)
                
            # Harvest results
            if hasattr(self.lattice_ref, 'oscillators'):
                # Extract phase coherence
                phases = np.array([o.phase for o in self.lattice_ref.oscillators])
                self.metrics.phase_coherence = np.abs(np.mean(np.exp(1j * phases)))
                
                # Count oscillations
                self.metrics.oscillations_detected = len(self.lattice_ref.oscillators)
                
            # Calculate efficiency
            self.metrics.efficiency_gain = self._calculate_efficiency_gain()
            
            burst_result = {
                'final_energy': self.metrics.energy_current,
                'phase_coherence': self.metrics.phase_coherence,
                'harvested_state': self._harvest_state()
            }
            
        finally:
            # Ensure token emission stops
            token_emission_task.cancel()
            try:
                await token_emission_task
            except asyncio.CancelledError:
                pass
                
        return burst_result
        
    async def _simulate_burst(self, intensity: float, duration: float) -> Dict[str, Any]:
        """Simulate burst for testing without actual lattice"""
        # Simulate energy decay
        start_time = time.time()
        initial_energy = 100.0
        
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            # Exponential decay with oscillations
            current_energy = initial_energy * np.exp(-0.1 * elapsed) * (1 + 0.1 * np.sin(10 * elapsed))
            self.metrics.energy_current = current_energy
            self.energy_history.append(current_energy)
            
            await asyncio.sleep(0.01)
            
        self.metrics.energy_harvested = initial_energy - current_energy
        self.metrics.efficiency_gain = 3.5 * intensity  # Simulated gain
        self.metrics.phase_coherence = 0.8
        
        return {
            'final_energy': current_energy,
            'phase_coherence': self.metrics.phase_coherence,
            'simulated': True
        }
        
    async def _emit_tokens(self, duration: float):
        """Emit observer tokens during burst"""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Create token
            token = ObserverToken(
                token_id=f"token_{self.token_counter}",
                timestamp=time.time(),
                energy_snapshot=self.metrics.energy_current,
                phase_state=self._capture_phase_state(),
                metadata={
                    'burst_progress': (time.time() - start_time) / duration,
                    'coherence': self.metrics.phase_coherence
                }
            )
            
            self.emitted_tokens.append(token)
            self.token_counter += 1
            self.metrics.tokens_emitted += 1
            
            # Call observer if registered
            if self.observer_callback:
                await self.observer_callback(token)
                
            # Wait for next emission
            await asyncio.sleep(1.0 / TOKEN_EMISSION_RATE)
            
    async def _cooldown_timer(self):
        """Handle cooldown period"""
        await asyncio.sleep(COOLDOWN_PERIOD)
        if self.state == ChannelState.COOLDOWN:
            self.state = ChannelState.IDLE
            logger.info("Cooldown complete, channel ready")
            
    def _calculate_system_energy(self) -> float:
        """Calculate total system energy"""
        if self.lattice_ref and hasattr(self.lattice_ref, 'oscillators'):
            # Sum oscillator energies
            total = 0.0
            for osc in self.lattice_ref.oscillators:
                # Energy = amplitude^2 + kinetic term
                total += osc.amplitude**2 + 0.5 * osc.natural_freq**2
            return total
        else:
            # Simulation mode
            return 100.0 if not self.energy_history else self.energy_history[-1]
            
    def _calculate_efficiency_gain(self) -> float:
        """Calculate chaos efficiency gain"""
        if len(self.energy_history) < 10:
            return 1.0
            
        # Look for energy recycling patterns
        energy_array = np.array(self.energy_history[-100:])
        
        # Fourier analysis to find oscillation modes
        fft = np.fft.fft(energy_array)
        freqs = np.fft.fftfreq(len(energy_array))
        
        # Count significant modes
        significant_modes = np.sum(np.abs(fft) > np.mean(np.abs(fft)) * 2)
        
        # Efficiency proportional to mode richness
        base_efficiency = 1.0 + 0.5 * significant_modes
        
        # Bonus for phase coherence
        coherence_bonus = self.metrics.phase_coherence * 2.0
        
        return base_efficiency + coherence_bonus
        
    def _capture_phase_state(self) -> np.ndarray:
        """Capture current phase state"""
        if self.lattice_ref and hasattr(self.lattice_ref, 'oscillators'):
            return np.array([o.phase for o in self.lattice_ref.oscillators])
        else:
            # Simulation
            return np.random.randn(10)
            
    def _harvest_state(self) -> Optional[np.ndarray]:
        """Harvest the post-burst lattice state"""
        if self.lattice_ref and hasattr(self.lattice_ref, 'oscillators'):
            # Create complex representation
            amplitudes = np.array([o.amplitude for o in self.lattice_ref.oscillators])
            phases = np.array([o.phase for o in self.lattice_ref.oscillators])
            return amplitudes * np.exp(1j * phases)
        return None
        
    def _persist_burst_results(self, result: Dict[str, Any]):
        """Persist burst results to torus registry"""
        if not self.registry:
            return
            
        try:
            # Store metrics
            self.registry.set(
                f"chaos_burst_{int(time.time())}",
                {
                    'metrics': self.metrics.__dict__,
                    'result': result,
                    'tokens': len(self.emitted_tokens)
                }
            )
        except Exception as e:
            logger.error(f"Failed to persist burst results: {e}")
            
    def get_status(self) -> Dict[str, Any]:
        """Get current controller status"""
        return {
            'state': self.state.value,
            'last_burst': self.last_burst_time,
            'cooldown_remaining': max(0, COOLDOWN_PERIOD - (time.time() - self.last_burst_time)),
            'total_tokens_emitted': self.token_counter,
            'current_metrics': self.metrics.__dict__,
            'emergency_stop': self.emergency_stop
        }
        
    def reset_emergency_stop(self):
        """Reset emergency stop flag"""
        self.emergency_stop = False
        if self.state == ChannelState.ERROR:
            self.state = ChannelState.IDLE
            
    async def spawn_soliton_from_harvest(self, harvested_state: np.ndarray) -> bool:
        """
        Spawn new soliton from harvested chaos energy
        
        Args:
            harvested_state: Complex state vector from burst
            
        Returns:
            Success flag
        """
        if self.lattice_ref is None:
            return False
            
        try:
            # Find location with maximum harvested amplitude
            max_idx = np.argmax(np.abs(harvested_state))
            amplitude = np.abs(harvested_state[max_idx])
            phase = np.angle(harvested_state[max_idx])
            
            # Add new oscillator at this configuration
            if hasattr(self.lattice_ref, 'add_oscillator'):
                from python.core.oscillator_lattice import SolitonPolarity
                
                # Determine polarity based on phase
                polarity = SolitonPolarity.DARK if phase < 0 else SolitonPolarity.BRIGHT
                
                # Add with harvested parameters
                self.lattice_ref.add_oscillator(
                    phase=phase,
                    amplitude=amplitude * 0.8,  # Slightly reduce to ensure stability
                    polarity=polarity,
                    natural_freq=0.1  # Low frequency for stability
                )
                
                logger.info(f"Spawned {polarity.value} soliton from harvested energy")
                return True
                
        except Exception as e:
            logger.error(f"Failed to spawn soliton: {e}")
            
        return False

# Test function
async def test_chaos_controller():
    """Test the chaos channel controller"""
    print("🌊 Testing Chaos Channel Controller")
    print("=" * 50)
    
    # Create controller
    controller = ChaosChannelController()
    
    # Test 1: Basic burst
    print("\n1️⃣ Testing basic chaos burst...")
    result = await controller.trigger(intensity=1.0, duration=2.0)
    print(f"Result: {result['success']}")
    if result['success']:
        print(f"Tokens emitted: {result['tokens_emitted']}")
        print(f"Efficiency gain: {result['efficiency_gain']:.2f}x")
        print(f"Energy harvested: {result['metrics']['energy_harvested']:.2f}")
        
    # Test 2: Cooldown check
    print("\n2️⃣ Testing cooldown...")
    result2 = await controller.trigger(intensity=1.0)
    print(f"Result: {result2['success']} (reason: {result2.get('reason')})")
    
    # Test 3: Status check
    print("\n3️⃣ Controller status:")
    status = controller.get_status()
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_chaos_controller())
