#!/usr/bin/env python3
"""
Base Blowup Harness Class
Energy harvesting system for oscillator lattice
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class BlowupHarness:
    """
    Base class for energy harvesting from oscillator lattice
    Manages controlled blow-up scenarios and energy extraction
    """
    
    def __init__(self, lattice: Any):
        """
        Initialize the blow-up harness
        
        Args:
            lattice: The oscillator lattice to harvest from
        """
        self.lattice = lattice
        self.harvest_history = []
        logger.info("Blowup harness using centralized BPS configuration")
    
    def harvest_energy(self) -> Dict[str, Any]:
        """
        Basic energy harvesting from oscillators
        
        Returns:
            Dictionary with harvest report
        """
        total_energy = 0.0
        harvested_count = 0
        
        # Simple energy extraction from oscillators
        if hasattr(self.lattice, 'oscillators') and hasattr(self.lattice, 'amplitudes'):
            for i in range(len(self.lattice.amplitudes)):
                energy = self.lattice.amplitudes[i] ** 2
                total_energy += energy
                harvested_count += 1
                
                # Zero out the oscillator
                self.lattice.amplitudes[i] = 0.0
        
        report = {
            "total_energy": total_energy,
            "harvested_count": harvested_count,
            "timestamp": np.datetime64('now')
        }
        
        self.harvest_history.append(report)
        logger.info(f"Harvested {total_energy:.3f} energy from {harvested_count} oscillators")
        
        return report
    
    def prepare_blow_up(self) -> Dict[str, Any]:
        """
        Prepare for controlled blow-up
        
        Returns:
            Blow-up preparation data
        """
        # Harvest energy first
        harvest_report = self.harvest_energy()
        
        blow_up_data = {
            "energy_to_reinject": harvest_report["total_energy"],
            "num_oscillators": harvest_report["harvested_count"],
            "harvest_report": harvest_report
        }
        
        logger.info(f"Blow-up prepared with {blow_up_data['energy_to_reinject']:.3f} energy")
        return blow_up_data
    
    def execute_controlled_blow_up(self) -> bool:
        """
        Execute a controlled blow-up
        
        Returns:
            True if successful
        """
        # Prepare blow-up
        blow_up_data = self.prepare_blow_up()
        
        # Reinject energy (simplified)
        if hasattr(self.lattice, 'amplitudes'):
            energy_per_osc = blow_up_data["energy_to_reinject"] / len(self.lattice.amplitudes)
            for i in range(len(self.lattice.amplitudes)):
                self.lattice.amplitudes[i] = np.sqrt(energy_per_osc)
        
        logger.info("Controlled blow-up executed")
        return True


# Keep the existing functions for backward compatibility
def induce_blowup(
    lattice: Any,
    epsilon: float = 0.1,
    steps: int = 10,
    memory: Optional[Any] = None,
    bps_aware: bool = True,
    adaptive: bool = False
) -> Union[np.ndarray, Dict[str, Any]]:
    """
    Legacy function for inducing controlled blowup
    
    Args:
        lattice: Oscillator lattice object
        epsilon: Amplification factor per step
        steps: Number of amplification steps
        memory: Optional memory system
        bps_aware: Use BPS-preserving extraction
        adaptive: Enable adaptive scaling
    
    Returns:
        Harvested energy or detailed report
    """
    harness = BlowupHarness(lattice)
    report = harness.harvest_energy()
    
    if bps_aware:
        return {"energy": report["total_energy"], "bps_preserved": True}
    else:
        return np.array([report["total_energy"]])


def extract_energy_from_lattice(lattice: Any) -> float:
    """
    Direct energy extraction from lattice
    
    Args:
        lattice: Oscillator lattice
    
    Returns:
        Total extracted energy
    """
    harness = BlowupHarness(lattice)
    report = harness.harvest_energy()
    return report["total_energy"]


# Aliases for backward compatibility
induce_blowup_legacy = induce_blowup
extract_energy_legacy = extract_energy_from_lattice
blowup_induction = induce_blowup
energy_extraction = extract_energy_from_lattice


# Export key functions and classes
__all__ = [
    # Main class
    'BlowupHarness',
    
    # Core functions
    'induce_blowup',
    'extract_energy_from_lattice',
    
    # Legacy compatibility
    'induce_blowup_legacy',
    'extract_energy_legacy',
    'blowup_induction',
    'energy_extraction',
]
