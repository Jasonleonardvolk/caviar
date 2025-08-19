"""
BPS-Enhanced Oscillator Module
Extends OscillatorLattice with BPS soliton support
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass
from enum import Enum

from python.core.oscillator_lattice import OscillatorLattice
from python.core.bps_config_enhanced import SolitonPolarity, BPS_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class Oscillator:
    """Individual oscillator with BPS support"""
    index: int
    phase: float
    amplitude: float
    frequency: float
    polarity: SolitonPolarity = SolitonPolarity.BRIGHT
    charge: float = 0.0  # Topological charge for BPS
    locked: bool = False  # Phase-locked state
    
    def step(self, coupling_input: float, dt: float):
        """Update oscillator state based on polarity"""
        if self.polarity == SolitonPolarity.BPS:
            # BPS oscillators maintain topological energy saturation
            # Energy E = |Q|
            target_amplitude = abs(self.charge)
            
            # Clamp amplitude to topological bound
            if abs(self.amplitude - target_amplitude) > BPS_CONFIG.bps_energy_tolerance:
                # Smooth damping toward target
                self.amplitude = self.amplitude * 0.9 + target_amplitude * 0.1
            
            # BPS oscillators don't get phase-pulled by neighbors
            # They only update by their natural frequency
            self.phase += self.frequency * dt
            
        elif self.polarity == SolitonPolarity.DARK:
            # Dark soliton - inverted coupling
            self.phase += self.frequency * dt - coupling_input * dt
            
        else:  # BRIGHT
            # Standard bright soliton
            self.phase += self.frequency * dt + coupling_input * dt
        
        # Keep phase in [0, 2π]
        self.phase = self.phase % (2 * np.pi)


class BPSEnhancedLattice(OscillatorLattice):
    """Oscillator lattice with full BPS soliton support"""
    
    def __init__(self, size: int = 64, coupling_strength: float = 0.1,
                 dt: float = 0.01, adjacency: Optional[np.ndarray] = None,
                 integrator: str = "euler"):
        super().__init__(size, coupling_strength, dt, adjacency, integrator)
        
        # Initialize oscillator objects with polarity support
        self.oscillator_objects: List[Oscillator] = []
        for i in range(size):
            osc = Oscillator(
                index=i,
                phase=self.oscillators[i],
                amplitude=self.amplitudes[i],
                frequency=self.frequencies[i],
                polarity=SolitonPolarity.BRIGHT
            )
            self.oscillator_objects.append(osc)
        
        # BPS-specific tracking
        self.bps_indices: Set[int] = set()
        self.total_charge: float = 0.0
        
        logger.info("🌀 BPS-Enhanced Lattice initialized")
    
    def create_bps_soliton(self, index: int, charge: float = 1.0,
                           phase: Optional[float] = None) -> bool:
        """Create a BPS soliton at the specified index"""
        if index < 0 or index >= self.size:
            logger.error(f"Invalid index {index} for BPS soliton")
            return False
        
        if BPS_CONFIG.max_bps_oscillators and len(self.bps_indices) >= BPS_CONFIG.max_bps_oscillators:
            logger.warning(f"Maximum BPS oscillators ({BPS_CONFIG.max_bps_oscillators}) reached")
            return False
        
        osc = self.oscillator_objects[index]
        osc.polarity = SolitonPolarity.BPS
        osc.charge = charge
        osc.amplitude = abs(charge)  # E = |Q|
        
        if phase is not None:
            osc.phase = phase
        
        self.bps_indices.add(index)
        self.total_charge += charge
        
        logger.info(f"✨ Created BPS soliton at index {index} with charge {charge}")
        return True
    
    def remove_bps_soliton(self, index: int, annihilate: bool = False) -> bool:
        """Remove or annihilate a BPS soliton"""
        if index not in self.bps_indices:
            return False
        
        osc = self.oscillator_objects[index]
        
        if annihilate and BPS_CONFIG.strict_bps_mode:
            # In strict mode, require proper annihilation
            # This would need an anti-soliton of opposite charge
            logger.warning("BPS annihilation requires opposite charge - not implemented")
            return False
        
        # Remove BPS status
        self.total_charge -= osc.charge
        osc.polarity = SolitonPolarity.BRIGHT
        osc.charge = 0.0
        self.bps_indices.remove(index)
        
        logger.info(f"Removed BPS soliton at index {index}")
        return True
    
    def _compute_coupling(self) -> np.ndarray:
        """Compute coupling with BPS-aware rules"""
        coupling = np.zeros(self.size)
        
        for i in range(self.size):
            osc_i = self.oscillator_objects[i]
            coupling_input = 0.0
            
            # Get neighbors (all-to-all or from adjacency)
            if self.adjacency is not None:
                neighbors = np.where(self.adjacency[i] > 0)[0]
                weights = self.adjacency[i, neighbors]
            else:
                neighbors = np.arange(self.size)
                neighbors = neighbors[neighbors != i]
                weights = np.ones(len(neighbors))
            
            for j, weight in zip(neighbors, weights):
                osc_j = self.oscillator_objects[j]
                
                # Phase difference
                phase_diff = osc_j.phase - osc_i.phase
                
                # Kuramoto coupling term
                contrib = self.coupling_strength * weight * osc_j.amplitude * np.sin(phase_diff)
                
                # Adjust based on neighbor's polarity
                if osc_j.polarity == SolitonPolarity.DARK:
                    contrib *= -1  # Dark soliton inverts influence
                elif osc_j.polarity == SolitonPolarity.BPS:
                    # BPS solitons have strong phase-locking influence
                    contrib *= BPS_CONFIG.bps_phase_lock_strength
                
                coupling_input += contrib
            
            # Apply polarity-specific rules for receiving oscillator
            if osc_i.polarity == SolitonPolarity.BPS:
                # BPS oscillators ignore external coupling
                coupling[i] = 0.0
            elif osc_i.polarity == SolitonPolarity.DARK:
                # Dark oscillators see inverted coupling
                coupling[i] = -coupling_input
            else:
                # Bright oscillators use normal coupling
                coupling[i] = coupling_input
        
        return coupling
    
    def step_enhanced(self):
        """Enhanced step with BPS support"""
        if not self.running:
            return
        
        # Compute coupling
        coupling = self._compute_coupling()
        
        # Update each oscillator
        for i, osc in enumerate(self.oscillator_objects):
            osc.step(coupling[i], self.dt)
            
            # Sync back to arrays for compatibility
            self.oscillators[i] = osc.phase
            self.amplitudes[i] = osc.amplitude
            self.frequencies[i] = osc.frequency
        
        # Check charge conservation if enabled
        if BPS_CONFIG.log_charge_conservation:
            current_charge = sum(osc.charge for osc in self.oscillator_objects 
                               if osc.polarity == SolitonPolarity.BPS)
            if abs(current_charge - self.total_charge) > BPS_CONFIG.charge_conservation_tolerance:
                logger.warning(f"Charge conservation violated: {current_charge} != {self.total_charge}")
    
    def get_bps_report(self) -> Dict:
        """Generate BPS soliton status report"""
        bps_oscillators = [osc for osc in self.oscillator_objects 
                          if osc.polarity == SolitonPolarity.BPS]
        
        report = {
            "num_bps_solitons": len(bps_oscillators),
            "total_charge": self.total_charge,
            "bps_indices": list(self.bps_indices),
            "energy_charge_compliance": []
        }
        
        for osc in bps_oscillators:
            energy = osc.amplitude ** 2  # Simplified energy
            target = osc.charge ** 2
            deviation = abs(energy - target)
            
            report["energy_charge_compliance"].append({
                "index": osc.index,
                "charge": osc.charge,
                "energy": energy,
                "deviation": deviation,
                "compliant": deviation < BPS_CONFIG.bps_energy_tolerance
            })
        
        return report
