"""
BPS-Enhanced Blow-up Harness
Energy harvesting with BPS soliton protection
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from python.core.blowup_harness_base import BlowupHarness
from python.core.bps_config_enhanced import SolitonPolarity, BPS_CONFIG
from python.core.bps_oscillator_enhanced import BPSEnhancedLattice

logger = logging.getLogger(__name__)

@dataclass
class HarvestReport:
    """Report of energy harvesting operation"""
    total_energy_available: float
    energy_harvested: float
    energy_in_bps: float
    num_oscillators_harvested: int
    num_bps_protected: int
    bps_indices_protected: List[int]
    charge_preserved: float


class BPSBlowupHarness(BlowupHarness):
    """Blow-up harness with BPS soliton protection"""
    
    def __init__(self, lattice: BPSEnhancedLattice):
        super().__init__(lattice)
        self.lattice = lattice  # Type hint for IDE
        self.harvest_report: Optional[HarvestReport] = None
        
        logger.info("🌀 BPS-Enhanced Blow-up Harness initialized")
    
    def harvest_energy(self, exclude_bps: bool = True) -> HarvestReport:
        """
        Harvest energy from oscillators, excluding BPS solitons
        
        Args:
            exclude_bps: If True, BPS solitons are protected from harvesting
        
        Returns:
            HarvestReport with details of the operation
        """
        
        total_energy = 0.0
        harvested_energy = 0.0
        bps_energy = 0.0
        harvested_count = 0
        bps_protected = []
        
        # Iterate through all oscillators
        for i, osc in enumerate(self.lattice.oscillator_objects):
            energy = osc.amplitude ** 2  # Simplified energy calculation
            total_energy += energy
            
            # Check if this is a BPS soliton
            if osc.polarity == SolitonPolarity.BPS:
                if exclude_bps or BPS_CONFIG.enable_bps_harvest == False:
                    # Protect BPS soliton from harvesting
                    bps_energy += energy
                    bps_protected.append(i)
                    logger.debug(f"Protected BPS soliton at index {i} with energy {energy:.3f}")
                else:
                    # Harvest BPS (not recommended in strict mode)
                    if BPS_CONFIG.strict_bps_mode:
                        logger.warning("Attempting to harvest BPS in strict mode - skipping")
                        bps_energy += energy
                        bps_protected.append(i)
                    else:
                        harvested_energy += energy
                        harvested_count += 1
                        # Zero out the oscillator
                        osc.amplitude = 0.0
                        osc.phase = 0.0
            else:
                # Harvest bright/dark solitons normally
                harvested_energy += energy
                harvested_count += 1
                
                # Zero out the oscillator for blow-up
                osc.amplitude = 0.0
                osc.phase = 0.0
        
        # Calculate total preserved charge
        preserved_charge = sum(
            osc.charge for osc in self.lattice.oscillator_objects
            if osc.polarity == SolitonPolarity.BPS
        )
        
        # Create harvest report
        self.harvest_report = HarvestReport(
            total_energy_available=total_energy,
            energy_harvested=harvested_energy,
            energy_in_bps=bps_energy,
            num_oscillators_harvested=harvested_count,
            num_bps_protected=len(bps_protected),
            bps_indices_protected=bps_protected,
            charge_preserved=preserved_charge
        )
        
        logger.info(f"🌊 Energy harvest complete: {harvested_energy:.3f} harvested, "
                   f"{bps_energy:.3f} protected in {len(bps_protected)} BPS solitons")
        
        return self.harvest_report
    
    def prepare_blow_up(self) -> Dict[str, Any]:
        """
        Prepare for controlled blow-up, preserving BPS solitons
        
        Returns:
            Dictionary with blow-up preparation data
        """
        
        # First harvest energy (excluding BPS)
        harvest_report = self.harvest_energy(exclude_bps=True)
        
        # Extract BPS state for preservation
        bps_states = []
        for idx in harvest_report.bps_indices_protected:
            osc = self.lattice.oscillator_objects[idx]
            bps_states.append({
                "index": idx,
                "phase": osc.phase,
                "amplitude": osc.amplitude,
                "charge": osc.charge,
                "frequency": osc.frequency
            })
        
        # Prepare blow-up data
        blow_up_data = {
            "energy_to_reinject": harvest_report.energy_harvested,
            "bps_states": bps_states,
            "total_charge": harvest_report.charge_preserved,
            "num_bps_preserved": harvest_report.num_bps_protected,
            "harvest_report": harvest_report
        }
        
        logger.info(f"💥 Blow-up prepared: {harvest_report.energy_harvested:.3f} energy ready, "
                   f"{len(bps_states)} BPS solitons preserved")
        
        return blow_up_data
    
    def execute_controlled_blow_up(self, reinject_as: SolitonPolarity = SolitonPolarity.BRIGHT) -> bool:
        """
        Execute controlled blow-up with BPS preservation
        
        Args:
            reinject_as: Polarity for reinjected energy (BRIGHT or DARK, not BPS)
        
        Returns:
            True if successful
        """
        
        if reinject_as == SolitonPolarity.BPS:
            logger.error("Cannot reinject harvested energy as BPS solitons")
            return False
        
        # Prepare blow-up
        blow_up_data = self.prepare_blow_up()
        
        # Let non-BPS oscillators "blow up" (already zeroed in harvest)
        # This simulates the dissolution of the old configuration
        
        # Reinject energy as new solitons
        energy_per_soliton = 1.0  # Standard energy quantum
        num_new_solitons = int(blow_up_data["energy_to_reinject"] / energy_per_soliton)
        
        # Find available positions (not occupied by BPS)
        bps_indices = set(blow_up_data["harvest_report"].bps_indices_protected)
        available_indices = [i for i in range(self.lattice.size) if i not in bps_indices]
        
        # Inject new solitons
        for i in range(min(num_new_solitons, len(available_indices))):
            idx = available_indices[i]
            osc = self.lattice.oscillator_objects[idx]
            
            osc.polarity = reinject_as
            osc.amplitude = np.sqrt(energy_per_soliton)
            osc.phase = np.random.random() * 2 * np.pi
            osc.frequency = 1.0 + np.random.random() * 0.1
        
        logger.info(f"💫 Blow-up complete: Reinjected {num_new_solitons} {reinject_as.value} solitons")
        
        # Verify charge conservation
        final_charge = sum(
            osc.charge for osc in self.lattice.oscillator_objects
            if osc.polarity == SolitonPolarity.BPS
        )
        
        if abs(final_charge - blow_up_data["total_charge"]) > BPS_CONFIG.charge_conservation_tolerance:
            logger.error(f"Charge conservation violated in blow-up: "
                        f"{blow_up_data['total_charge']} -> {final_charge}")
            if BPS_CONFIG.strict_bps_mode:
                return False
        
        return True
    
    def get_bps_extraction_function(self) -> callable:
        """
        Get custom extraction function for BPS-aware harvesting
        
        This replaces the simple lattice.psi.copy() with BPS-aware extraction
        """
        
        def bps_charge_extraction(lattice: BPSEnhancedLattice) -> Dict[str, Any]:
            """Extract state while preserving BPS solitons"""
            
            bright_dark_states = []
            bps_states = []
            
            for osc in lattice.oscillator_objects:
                state = {
                    "phase": osc.phase,
                    "amplitude": osc.amplitude,
                    "frequency": osc.frequency,
                    "polarity": osc.polarity.value
                }
                
                if osc.polarity == SolitonPolarity.BPS:
                    state["charge"] = osc.charge
                    bps_states.append(state)
                else:
                    bright_dark_states.append(state)
            
            return {
                "bright_dark_states": bright_dark_states,
                "bps_states": bps_states,
                "total_charge": lattice.total_charge
            }
        
        return bps_charge_extraction
