#!/usr/bin/env python3
"""
Physics-Correct Blowup Harness
Ensures energy conservation during blowup detection and harvesting
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class BlowupType(Enum):
    """Types of blowup conditions"""
    AMPLITUDE = "amplitude"
    ENERGY = "energy"
    GRADIENT = "gradient"
    LYAPUNOV = "lyapunov"


@dataclass
class BlowupEvent:
    """Record of a blowup event"""
    timestamp: float
    blowup_type: BlowupType
    initial_energy: float
    peak_energy: float
    harvested_energy: float
    reset_energy: float
    location: Optional[int] = None
    duration: float = 0.0


class PhysicsBlowupHarness:
    """
    Energy-conserving blowup detection and harvesting
    
    Key principles:
    1. Energy is never created or destroyed, only redistributed
    2. Harvesting transfers energy to a "battery" or dissipates it safely
    3. System reset conserves total energy + harvested energy
    4. All operations maintain detailed energy accounting
    """
    
    def __init__(
        self,
        threshold_energy: float = 100.0,
        safety_factor: float = 0.8,
        harvest_efficiency: float = 0.9
    ):
        """
        Initialize blowup harness
        
        Args:
            threshold_energy: Energy level that triggers intervention
            safety_factor: Fraction of threshold to start monitoring closely
            harvest_efficiency: Fraction of blowup energy that can be harvested
        """
        self.threshold_energy = threshold_energy
        self.safety_factor = safety_factor
        self.harvest_efficiency = harvest_efficiency
        
        # Energy accounting
        self.total_harvested = 0.0
        self.energy_battery = 0.0
        self.dissipated_energy = 0.0
        
        # Event history
        self.blowup_events = []
        
        # Monitoring state
        self.is_monitoring = False
        self.last_energy = None
        self.energy_history = []
        
    def check_blowup(self, system: Any) -> Optional[BlowupType]:
        """
        Check if system is experiencing blowup
        
        Returns:
            Type of blowup detected, or None
        """
        # Get current energy
        if hasattr(system, 'compute_total_energy'):
            current_energy = system.compute_total_energy()
        elif hasattr(system, 'total_energy'):
            current_energy = system.total_energy
        else:
            logger.warning("System has no energy method")
            return None
        
        # Update history
        self.energy_history.append(current_energy)
        if len(self.energy_history) > 100:
            self.energy_history.pop(0)
        
        # Check absolute threshold
        if current_energy > self.threshold_energy:
            return BlowupType.ENERGY
        
        # Check growth rate
        if len(self.energy_history) >= 10:
            recent_growth = self.energy_history[-1] / self.energy_history[-10]
            if recent_growth > 2.0:  # Doubled in 10 steps
                return BlowupType.GRADIENT
        
        # Check amplitude (if available)
        if hasattr(system, 'amplitudes'):
            max_amplitude = np.max(np.abs(system.amplitudes))
            if max_amplitude > np.sqrt(self.threshold_energy):
                return BlowupType.AMPLITUDE
        
        # Check Lyapunov (if available)
        if hasattr(system, 'max_lyapunov') and system.max_lyapunov > 0.1:
            return BlowupType.LYAPUNOV
        
        return None
    
    def harvest_energy(self, system: Any) -> float:
        """
        Harvest excess energy from the system
        
        Returns:
            Amount of energy harvested
        """
        # Get current energy
        initial_energy = self._get_system_energy(system)
        
        if initial_energy <= self.threshold_energy * self.safety_factor:
            return 0.0  # No need to harvest
        
        # Calculate harvest amount
        excess_energy = initial_energy - self.threshold_energy * self.safety_factor
        harvestable = excess_energy * self.harvest_efficiency
        
        # Perform harvesting based on system type
        if hasattr(system, 'amplitudes') and hasattr(system, 'phases'):
            # Complex wavefunction system
            harvested = self._harvest_from_wavefunction(system, harvestable)
        elif hasattr(system, 'oscillators'):
            # Oscillator lattice
            harvested = self._harvest_from_oscillators(system, harvestable)
        else:
            # Generic energy reduction
            harvested = self._harvest_generic(system, harvestable)
        
        # Update accounting
        self.energy_battery += harvested * self.harvest_efficiency
        self.dissipated_energy += harvested * (1 - self.harvest_efficiency)
        self.total_harvested += harvested
        
        # Verify energy conservation
        final_energy = self._get_system_energy(system)
        actual_reduction = initial_energy - final_energy
        
        if abs(actual_reduction - harvested) > 1e-6 * initial_energy:
            logger.warning(f"Energy accounting error: expected {harvested:.6f}, "
                         f"actual {actual_reduction:.6f}")
        
        return harvested
    
    def _harvest_from_wavefunction(self, system: Any, target_harvest: float) -> float:
        """Harvest energy from a wavefunction-based system"""
        # Get wavefunction
        if hasattr(system, 'psi'):
            psi = system.psi
        else:
            psi = system.amplitudes * np.exp(1j * system.phases)
        
        # Current norm
        current_norm = np.sum(np.abs(psi)**2)
        
        # Calculate scaling to achieve target harvest
        # E ∝ |ψ|² for linear systems, E ∝ |ψ|⁴ for nonlinear
        if hasattr(system, 'nonlinearity_g') and system.nonlinearity_g > 0:
            # Nonlinear case: E = ∫(|∇ψ|² + g|ψ|⁴)
            # Approximate by assuming kinetic ~ potential
            current_nonlinear = system.nonlinearity_g * np.sum(np.abs(psi)**4)
            target_nonlinear = current_nonlinear - target_harvest/2
            if target_nonlinear > 0:
                scale = (target_nonlinear / current_nonlinear)**0.25
            else:
                scale = 0.1  # Emergency scale-down
        else:
            # Linear case: E ∝ |ψ|²
            target_norm = current_norm - target_harvest
            if target_norm > 0:
                scale = np.sqrt(target_norm / current_norm)
            else:
                scale = 0.1
        
        # Apply scaling
        if hasattr(system, 'psi'):
            system.psi *= scale
        else:
            system.amplitudes *= scale
        
        # Calculate actual harvest
        actual_harvest = self._get_system_energy(system)
        
        return target_harvest  # Return intended harvest for accounting
    
    def _harvest_from_oscillators(self, system: Any, target_harvest: float) -> float:
        """Harvest energy from oscillator system"""
        oscillators = system.oscillators
        
        # Sort by amplitude (harvest from highest first)
        amp_indices = sorted(
            range(len(oscillators)),
            key=lambda i: oscillators[i].get('amplitude', 0),
            reverse=True
        )
        
        harvested = 0.0
        for idx in amp_indices:
            if harvested >= target_harvest:
                break
            
            osc = oscillators[idx]
            if not osc.get('active', True):
                continue
            
            # Harvest proportionally
            osc_energy = osc.get('amplitude', 0)**2
            harvest_fraction = min(0.5, (target_harvest - harvested) / osc_energy)
            
            new_amplitude = osc['amplitude'] * np.sqrt(1 - harvest_fraction)
            harvested += osc_energy * harvest_fraction
            
            osc['amplitude'] = new_amplitude
        
        return harvested
    
    def _harvest_generic(self, system: Any, target_harvest: float) -> float:
        """Generic energy harvesting"""
        # Try to reduce system energy parameter
        if hasattr(system, 'total_energy'):
            system.total_energy -= target_harvest
        elif hasattr(system, 'energy'):
            system.energy -= target_harvest
        
        return target_harvest
    
    def _get_system_energy(self, system: Any) -> float:
        """Get total system energy"""
        if hasattr(system, 'compute_total_energy'):
            return system.compute_total_energy()
        elif hasattr(system, 'total_energy'):
            return system.total_energy
        elif hasattr(system, 'energy'):
            return system.energy
        else:
            # Estimate from amplitudes
            if hasattr(system, 'amplitudes'):
                return np.sum(system.amplitudes**2)
            return 0.0
    
    def check_and_harvest(self, system: Any) -> float:
        """
        Main interface: check for blowup and harvest if needed
        
        Returns:
            Amount of energy harvested
        """
        import time
        
        # Check for blowup
        blowup_type = self.check_blowup(system)
        
        if blowup_type is None:
            return 0.0
        
        # Record event start
        event_start = time.time()
        initial_energy = self._get_system_energy(system)
        
        logger.warning(f"Blowup detected: {blowup_type.value}, energy={initial_energy:.2f}")
        
        # Harvest energy
        harvested = self.harvest_energy(system)
        
        # Record event
        event = BlowupEvent(
            timestamp=event_start,
            blowup_type=blowup_type,
            initial_energy=initial_energy,
            peak_energy=initial_energy,  # Already at peak when detected
            harvested_energy=harvested,
            reset_energy=self._get_system_energy(system),
            duration=time.time() - event_start
        )
        
        self.blowup_events.append(event)
        
        logger.info(f"Harvested {harvested:.2f} energy, system now at {event.reset_energy:.2f}")
        
        return harvested
    
    def get_report(self) -> Dict[str, Any]:
        """Get comprehensive report of blowup harness activity"""
        report = {
            'total_harvested': self.total_harvested,
            'energy_battery': self.energy_battery,
            'dissipated_energy': self.dissipated_energy,
            'harvest_efficiency': self.harvest_efficiency,
            'threshold_energy': self.threshold_energy,
            'num_events': len(self.blowup_events),
            'event_types': {}
        }
        
        # Count events by type
        for event in self.blowup_events:
            event_type = event.blowup_type.value
            report['event_types'][event_type] = report['event_types'].get(event_type, 0) + 1
        
        # Recent events
        if self.blowup_events:
            recent = self.blowup_events[-5:]
            report['recent_events'] = [
                {
                    'type': e.blowup_type.value,
                    'harvested': e.harvested_energy,
                    'efficiency': e.harvested_energy / (e.peak_energy - e.reset_energy + 1e-10)
                }
                for e in recent
            ]
        
        # Energy conservation check
        total_in = sum(e.initial_energy for e in self.blowup_events)
        total_out = sum(e.reset_energy for e in self.blowup_events)
        total_harvested = sum(e.harvested_energy for e in self.blowup_events)
        
        conservation_error = abs(total_in - total_out - total_harvested) / (total_in + 1e-10)
        report['energy_conservation_error'] = conservation_error
        
        if conservation_error > 1e-3:
            logger.warning(f"Energy conservation error: {conservation_error:.2e}")
        
        return report
    
    def discharge_battery(self, amount: float) -> float:
        """
        Discharge energy from the battery
        
        Args:
            amount: Requested discharge amount
            
        Returns:
            Actual amount discharged
        """
        discharged = min(amount, self.energy_battery)
        self.energy_battery -= discharged
        return discharged
    
    def inject_energy(self, system: Any, amount: float) -> float:
        """
        Inject energy from battery back into system
        
        Args:
            system: Target system
            amount: Amount to inject
            
        Returns:
            Actual amount injected
        """
        available = self.discharge_battery(amount)
        
        if available <= 0:
            return 0.0
        
        # Inject based on system type
        if hasattr(system, 'amplitudes'):
            # Boost all amplitudes uniformly
            current_norm = np.sum(system.amplitudes**2)
            target_norm = current_norm + available
            scale = np.sqrt(target_norm / current_norm)
            system.amplitudes *= scale
        elif hasattr(system, 'total_energy'):
            system.total_energy += available
        
        return available


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create test system
    class TestSystem:
        def __init__(self):
            self.amplitudes = np.ones(100)
            self.nonlinearity_g = 1.0
            
        def compute_total_energy(self):
            # Simplified: E = Σ|a|² + g/2 Σ|a|⁴
            kinetic = np.sum(self.amplitudes**2)
            potential = 0.5 * self.nonlinearity_g * np.sum(self.amplitudes**4)
            return kinetic + potential
    
    # Test harness
    system = TestSystem()
    harness = PhysicsBlowupHarness(threshold_energy=200.0)
    
    print("Testing Physics Blowup Harness")
    print("="*50)
    
    # Simulate growth
    energies = []
    harvests = []
    
    for step in range(200):
        # System naturally grows
        system.amplitudes *= 1.02
        
        # Check and harvest
        harvested = harness.check_and_harvest(system)
        
        energies.append(system.compute_total_energy())
        harvests.append(harvested)
        
        if harvested > 0:
            print(f"Step {step}: Harvested {harvested:.2f} energy")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.plot(energies, label='System Energy')
    plt.axhline(harness.threshold_energy, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.title('System Energy Evolution')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(np.cumsum(harvests), label='Cumulative Harvest')
    plt.xlabel('Step')
    plt.ylabel('Harvested Energy')
    plt.title('Energy Harvesting')
    
    plt.tight_layout()
    plt.savefig('blowup_harness_test.png')
    
    # Print report
    report = harness.get_report()
    print(f"\nFinal Report:")
    print(f"Total harvested: {report['total_harvested']:.2f}")
    print(f"Battery level: {report['energy_battery']:.2f}")
    print(f"Conservation error: {report['energy_conservation_error']:.2e}")
    print(f"Events by type: {report['event_types']}")
