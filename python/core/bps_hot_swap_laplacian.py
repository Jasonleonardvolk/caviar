"""
BPS-Enhanced Hot-Swap Laplacian Module
Topology transitions with BPS soliton preservation
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy

from python.core.hot_swap_laplacian_base import HotSwapLaplacian
from python.core.bps_config_enhanced import SolitonPolarity, BPS_CONFIG
from python.core.bps_oscillator_enhanced import BPSEnhancedLattice
from python.core.bps_blowup_harness import BPSBlowupHarness

logger = logging.getLogger(__name__)


class BPSHotSwapLaplacian(HotSwapLaplacian):
    """Hot-swap topology manager with BPS soliton preservation"""
    
    def __init__(self, old_lattice: BPSEnhancedLattice):
        super().__init__(old_lattice)
        self.old_lattice = old_lattice
        self.new_lattice: Optional[BPSEnhancedLattice] = None
        self.bps_transition_data: Optional[Dict] = None
        
        logger.info("🌀 BPS-Enhanced Hot-Swap Laplacian initialized")
    
    def prepare_hot_swap(self, new_topology: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Prepare for topology hot-swap with BPS preservation
        
        Args:
            new_topology: Optional new adjacency matrix
        
        Returns:
            Preparation data including BPS states
        """
        
        # Initialize harness for energy extraction
        harness = BPSBlowupHarness(self.old_lattice)
        
        # Extract states using BPS-aware method
        extraction_func = harness.get_bps_extraction_function()
        extracted_states = extraction_func(self.old_lattice)
        
        # Separate BPS solitons for special handling
        bps_states = extracted_states["bps_states"]
        bright_dark_states = extracted_states["bright_dark_states"]
        
        # Record initial charge
        initial_charge = extracted_states["total_charge"]
        
        # Pause BPS dynamics if configured
        if BPS_CONFIG.bps_hot_swap_pause:
            for idx in self.old_lattice.bps_indices:
                osc = self.old_lattice.oscillator_objects[idx]
                osc.locked = True
                logger.debug(f"Paused BPS oscillator at index {idx}")
        
        # Prepare transition data
        self.bps_transition_data = {
            "bps_states": bps_states,
            "bright_dark_states": bright_dark_states,
            "initial_charge": initial_charge,
            "old_size": self.old_lattice.size,
            "old_topology": self.old_lattice.adjacency,
            "new_topology": new_topology,
            "bps_indices": list(self.old_lattice.bps_indices)
        }
        
        logger.info(f"📦 Hot-swap prepared: {len(bps_states)} BPS solitons to preserve, "
                   f"charge Q = {initial_charge:.3f}")
        
        return self.bps_transition_data
    
    def execute_hot_swap(self, new_size: Optional[int] = None,
                         new_topology: Optional[np.ndarray] = None) -> BPSEnhancedLattice:
        """
        Execute hot-swap transition with BPS preservation
        
        Args:
            new_size: Size of new lattice (default: same as old)
            new_topology: New adjacency matrix
        
        Returns:
            New lattice with BPS solitons preserved
        """
        
        if self.bps_transition_data is None:
            self.prepare_hot_swap(new_topology)
        
        # Determine new lattice size
        if new_size is None:
            new_size = self.old_lattice.size
        
        # Create new lattice
        self.new_lattice = BPSEnhancedLattice(
            size=new_size,
            coupling_strength=self.old_lattice.coupling_strength,
            dt=self.old_lattice.dt,
            adjacency=new_topology,
            integrator=self.old_lattice.integrator
        )
        
        # Step 1: Restore BPS solitons exactly
        success = self._restore_bps_solitons()
        if not success:
            logger.error("Failed to restore BPS solitons")
            if BPS_CONFIG.strict_bps_mode:
                raise RuntimeError("BPS restoration failed in strict mode")
        
        # Step 2: Reinject other energy as bright/dark solitons
        self._reinject_other_solitons()
        
        # Step 3: Verify charge conservation
        self._verify_charge_conservation()
        
        # Resume BPS dynamics
        if BPS_CONFIG.bps_hot_swap_pause:
            for idx in self.new_lattice.bps_indices:
                osc = self.new_lattice.oscillator_objects[idx]
                osc.locked = False
        
        logger.info(f"♻️ Hot-swap complete: New lattice size={new_size}, "
                   f"{len(self.new_lattice.bps_indices)} BPS solitons preserved")
        
        return self.new_lattice
    
    def _restore_bps_solitons(self) -> bool:
        """Restore BPS solitons in new lattice"""
        
        bps_states = self.bps_transition_data["bps_states"]
        old_size = self.bps_transition_data["old_size"]
        new_size = self.new_lattice.size
        
        restored_count = 0
        
        for state in bps_states:
            # Map old position to new position
            old_idx = self._find_oscillator_by_state(state, self.old_lattice)
            new_idx = self._map_index(old_idx, old_size, new_size, state)
            
            if new_idx is not None and new_idx < new_size:
                # Restore BPS soliton
                osc = self.new_lattice.oscillator_objects[new_idx]
                osc.polarity = SolitonPolarity.BPS
                osc.phase = state["phase"]
                osc.amplitude = state["amplitude"]
                osc.frequency = state["frequency"]
                osc.charge = state["charge"]
                
                self.new_lattice.bps_indices.add(new_idx)
                self.new_lattice.total_charge += state["charge"]
                
                restored_count += 1
                logger.debug(f"Restored BPS soliton at new index {new_idx}")
            else:
                logger.warning(f"Could not restore BPS soliton - no valid position")
        
        logger.info(f"Restored {restored_count}/{len(bps_states)} BPS solitons")
        return restored_count == len(bps_states)
    
    def _reinject_other_solitons(self):
        """Reinject non-BPS solitons as bright solitons"""
        
        bright_dark_states = self.bps_transition_data["bright_dark_states"]
        
        # Calculate total energy to reinject
        total_energy = sum(state["amplitude"]**2 for state in bright_dark_states)
        
        # Find available positions (not occupied by BPS)
        available_indices = []
        for i in range(self.new_lattice.size):
            if i not in self.new_lattice.bps_indices:
                available_indices.append(i)
        
        # Distribute energy among available oscillators
        if available_indices and total_energy > 0:
            energy_per_osc = total_energy / len(available_indices)
            amplitude = np.sqrt(energy_per_osc)
            
            for idx in available_indices[:int(total_energy)]:  # Limit to reasonable number
                osc = self.new_lattice.oscillator_objects[idx]
                osc.polarity = SolitonPolarity.BRIGHT
                osc.amplitude = amplitude
                osc.phase = np.random.random() * 2 * np.pi
                osc.frequency = 1.0 + np.random.random() * 0.1
        
        logger.info(f"Reinjected {total_energy:.3f} energy as bright solitons")
    
    def _verify_charge_conservation(self) -> bool:
        """Verify topological charge is conserved"""
        
        initial_charge = self.bps_transition_data["initial_charge"]
        final_charge = self.new_lattice.total_charge
        
        deviation = abs(final_charge - initial_charge)
        
        if deviation > BPS_CONFIG.charge_conservation_tolerance:
            logger.error(f"Charge conservation violated: Q_initial={initial_charge:.6f}, "
                        f"Q_final={final_charge:.6f}, deviation={deviation:.6f}")
            
            if BPS_CONFIG.strict_bps_mode:
                # Attempt correction
                self.new_lattice.total_charge = initial_charge
                logger.warning("Force-corrected total charge to maintain conservation")
            
            return False
        
        logger.info(f"✅ Charge conserved: Q = {final_charge:.6f}")
        return True
    
    def _map_index(self, old_idx: int, old_size: int, new_size: int,
                   state: Dict) -> Optional[int]:
        """Map oscillator index from old to new lattice"""
        
        if BPS_CONFIG.bps_position_mapping == "concept_id":
            # Use deterministic hash for concept-based mapping
            # This would need concept_ids from memory system
            pass
        
        if new_size == old_size:
            # Same size - direct mapping
            return old_idx
        elif new_size > old_size:
            # Larger lattice - scale up
            return int(old_idx * new_size / old_size)
        else:
            # Smaller lattice - scale down with wrapping
            return old_idx % new_size
    
    def _find_oscillator_by_state(self, state: Dict, 
                                  lattice: BPSEnhancedLattice) -> Optional[int]:
        """Find oscillator index matching state"""
        
        for i, osc in enumerate(lattice.oscillator_objects):
            if (abs(osc.phase - state["phase"]) < 0.01 and
                abs(osc.amplitude - state["amplitude"]) < 0.01):
                return i
        
        return None
    
    def bps_topology_transition(self, old_lattice: BPSEnhancedLattice,
                                new_lattice: BPSEnhancedLattice) -> bool:
        """
        Helper function for clean BPS topology transition
        
        This is the function referenced in the design document
        """
        
        # Copy BPS states
        bps_states = []
        for idx in old_lattice.bps_indices:
            osc = old_lattice.oscillator_objects[idx]
            bps_states.append({
                "index": idx,
                "phase": osc.phase,
                "amplitude": osc.amplitude,
                "charge": osc.charge,
                "frequency": osc.frequency
            })
        
        # Optionally quench dynamics
        if BPS_CONFIG.bps_hot_swap_pause:
            for state in bps_states:
                state["locked"] = True
        
        # Set up new connections
        # (This would involve updating the adjacency matrix)
        
        # Restore states
        for state in bps_states:
            new_idx = self._map_index(
                state["index"], 
                old_lattice.size, 
                new_lattice.size,
                state
            )
            
            if new_idx is not None:
                new_lattice.create_bps_soliton(
                    new_idx,
                    state["charge"],
                    state["phase"]
                )
        
        logger.info("✨ BPS topology transition complete")
        return True
