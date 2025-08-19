"""
BPS-Enhanced Soliton Memory Integration
Extends soliton memory with full BPS soliton support
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json

from python.core.soliton_memory_integration import (
    EnhancedSolitonMemory, SolitonMemoryEntry, 
    MemoryType, VaultStatus
)
from python.core.bps_config_enhanced import SolitonPolarity, BPS_CONFIG
from python.core.bps_oscillator_enhanced import BPSEnhancedLattice

logger = logging.getLogger(__name__)

@dataclass
class BPSSolitonMemoryEntry(SolitonMemoryEntry):
    """Extended memory entry with BPS support"""
    polarity: SolitonPolarity = SolitonPolarity.BRIGHT
    charge: float = 1.0  # Topological charge for BPS
    oscillator_index: Optional[int] = None  # Lattice position
    energy_locked: bool = False  # E = |Q| enforcement
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with BPS fields"""
        base_dict = {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "phase": self.phase,
            "amplitude": self.amplitude,
            "frequency": self.frequency,
            "timestamp": self.timestamp.isoformat(),
            "concept_ids": self.concept_ids,
            "sources": self.sources,
            "vault_status": self.vault_status.value,
            "polarity": self.polarity.value,
            "charge": self.charge,
            "oscillator_index": self.oscillator_index,
            "energy_locked": self.energy_locked,
            "metadata": self.metadata
        }
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BPSSolitonMemoryEntry':
        """Create from dictionary with BPS fields"""
        # Parse enums
        memory_type = MemoryType(data.get("memory_type", "semantic"))
        vault_status = VaultStatus(data.get("vault_status", "active"))
        polarity = SolitonPolarity(data.get("polarity", "bright"))
        
        # Parse timestamp
        timestamp = datetime.fromisoformat(data["timestamp"])
        
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=memory_type,
            phase=data["phase"],
            amplitude=data["amplitude"],
            frequency=data["frequency"],
            timestamp=timestamp,
            concept_ids=data["concept_ids"],
            sources=data["sources"],
            vault_status=vault_status,
            polarity=polarity,
            charge=data.get("charge", 1.0),
            oscillator_index=data.get("oscillator_index"),
            energy_locked=data.get("energy_locked", False),
            metadata=data.get("metadata", {})
        )


class BPSEnhancedSolitonMemory(EnhancedSolitonMemory):
    """Soliton memory system with full BPS support"""
    
    def __init__(self, lattice: Optional[BPSEnhancedLattice] = None):
        super().__init__()
        self.lattice = lattice
        self.bps_memories: Dict[str, BPSSolitonMemoryEntry] = {}
        self.charge_registry: Dict[int, float] = {}  # oscillator_index -> charge
        
        logger.info("🌀 BPS-Enhanced Soliton Memory initialized")
    
    def store_bps_soliton(self, 
                          content: str,
                          concept_ids: List[str],
                          charge: float = 1.0,
                          sources: Optional[List[str]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a BPS soliton memory"""
        
        # Generate memory ID
        memory_id = self._generate_memory_id(content, concept_ids)
        
        # Calculate phase from concepts
        phase = self._calculate_concept_phase(concept_ids)
        
        # Find available oscillator
        oscillator_index = self._find_available_oscillator()
        
        if oscillator_index is None:
            logger.error("No available oscillator for BPS soliton")
            return None
        
        # Create BPS memory entry
        entry = BPSSolitonMemoryEntry(
            id=memory_id,
            content=content,
            memory_type=MemoryType.SEMANTIC,  # BPS are typically semantic
            phase=phase,
            amplitude=abs(charge),  # E = |Q|
            frequency=1.0,  # Standard frequency for BPS
            timestamp=datetime.now(timezone.utc),
            concept_ids=concept_ids,
            sources=sources or [],
            polarity=SolitonPolarity.BPS,
            charge=charge,
            oscillator_index=oscillator_index,
            energy_locked=True,
            metadata=metadata or {}
        )
        
        # Store in memory system
        self.bps_memories[memory_id] = entry
        self.charge_registry[oscillator_index] = charge
        
        # Create BPS soliton in lattice if available
        if self.lattice:
            success = self.lattice.create_bps_soliton(
                oscillator_index, charge, phase
            )
            if not success:
                logger.error(f"Failed to create BPS soliton in lattice at index {oscillator_index}")
                del self.bps_memories[memory_id]
                del self.charge_registry[oscillator_index]
                return None
        
        logger.info(f"✨ Stored BPS soliton memory {memory_id} with charge {charge}")
        return memory_id
    
    def retrieve_bps_memory(self, memory_id: str) -> Optional[BPSSolitonMemoryEntry]:
        """Retrieve a BPS soliton memory"""
        return self.bps_memories.get(memory_id)
    
    def get_total_topological_charge(self) -> float:
        """Calculate total topological charge"""
        total_charge = sum(entry.charge for entry in self.bps_memories.values())
        return total_charge
    
    def verify_charge_conservation(self) -> bool:
        """Verify topological charge is conserved"""
        memory_charge = self.get_total_topological_charge()
        
        if self.lattice:
            lattice_charge = self.lattice.total_charge
            deviation = abs(memory_charge - lattice_charge)
            
            if deviation > BPS_CONFIG.charge_conservation_tolerance:
                logger.error(f"Charge conservation violated: memory={memory_charge}, lattice={lattice_charge}")
                return False
        
        return True
    
    def _find_available_oscillator(self) -> Optional[int]:
        """Find an available oscillator index for new BPS soliton"""
        if not self.lattice:
            return None
        
        # Find unused oscillator
        used_indices = set(self.charge_registry.keys())
        
        for i in range(self.lattice.size):
            if i not in used_indices:
                return i
        
        return None
    
    def prepare_for_hot_swap(self) -> Dict[str, Any]:
        """Prepare BPS solitons for topology hot-swap"""
        swap_data = {
            "bps_memories": [],
            "total_charge": self.get_total_topological_charge(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Serialize all BPS memories
        for entry in self.bps_memories.values():
            swap_data["bps_memories"].append(entry.to_dict())
        
        # Pause BPS dynamics if configured
        if BPS_CONFIG.bps_hot_swap_pause and self.lattice:
            for idx in self.lattice.bps_indices:
                osc = self.lattice.oscillator_objects[idx]
                osc.locked = True
        
        logger.info(f"📦 Prepared {len(self.bps_memories)} BPS solitons for hot-swap")
        return swap_data
    
    def restore_after_hot_swap(self, swap_data: Dict[str, Any], 
                               new_lattice: Optional[BPSEnhancedLattice] = None) -> bool:
        """Restore BPS solitons after topology hot-swap"""
        
        if new_lattice:
            self.lattice = new_lattice
        
        if not self.lattice:
            logger.error("No lattice available for BPS restoration")
            return False
        
        # Clear current state
        self.bps_memories.clear()
        self.charge_registry.clear()
        
        # Restore each BPS memory
        for memory_data in swap_data["bps_memories"]:
            entry = BPSSolitonMemoryEntry.from_dict(memory_data)
            
            # Find new position (may differ if lattice size changed)
            new_index = self._map_oscillator_index(
                entry.oscillator_index, 
                entry.concept_ids
            )
            
            if new_index is not None:
                entry.oscillator_index = new_index
                self.bps_memories[entry.id] = entry
                self.charge_registry[new_index] = entry.charge
                
                # Recreate in lattice
                self.lattice.create_bps_soliton(
                    new_index, 
                    entry.charge, 
                    entry.phase
                )
        
        # Verify charge conservation
        expected_charge = swap_data["total_charge"]
        actual_charge = self.get_total_topological_charge()
        
        if abs(expected_charge - actual_charge) > BPS_CONFIG.charge_conservation_tolerance:
            logger.error(f"Charge not conserved in hot-swap: {expected_charge} -> {actual_charge}")
            if BPS_CONFIG.strict_bps_mode:
                return False
        
        logger.info(f"♻️ Restored {len(self.bps_memories)} BPS solitons after hot-swap")
        return True
    
    def _map_oscillator_index(self, old_index: int, 
                              concept_ids: List[str]) -> Optional[int]:
        """Map oscillator index from old to new lattice"""
        
        if BPS_CONFIG.bps_position_mapping == "concept_id":
            # Use concept ID hash for consistent positioning
            if concept_ids:
                hash_val = hash(concept_ids[0]) % self.lattice.size
                if hash_val not in self.charge_registry:
                    return hash_val
        
        # Fall back to finding nearest available
        return self._find_available_oscillator()
    
    def get_bps_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics for BPS soliton system"""
        metrics = {
            "num_bps_solitons": len(self.bps_memories),
            "total_charge": self.get_total_topological_charge(),
            "charge_conserved": self.verify_charge_conservation(),
            "energy_compliance": []
        }
        
        if self.lattice:
            lattice_report = self.lattice.get_bps_report()
            metrics.update(lattice_report)
        
        return metrics
