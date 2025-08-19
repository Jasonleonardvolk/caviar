#!/usr/bin/env python3
"""
Dark Soliton Integration for Python Memory System
Adds polarity field and dark soliton handling to memory entries
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

# Patch for soliton_memory_integration.py

# Add polarity field to SolitonMemoryEntry
@dataclass 
class SolitonMemoryEntryPatched:
    """Enhanced memory entry with dark soliton support"""
    # ... existing fields ...
    polarity: str = "bright"  # NEW: "bright" or "dark" soliton type
    baseline_oscillator_idx: Optional[int] = None  # NEW: For dark solitons

# Enhancement for EnhancedSolitonMemory class
class DarkSolitonMemoryEnhancements:
    """Mixin for dark soliton functionality"""
    
    def store_dark_memory(
        self,
        content: str,
        concept_ids: List[str],
        reason: str = "suppression",
        sources: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Store a dark soliton memory to suppress/cancel a concept"""
        
        # Calculate phase for the concept
        phase = self._calculate_concept_phase(concept_ids)
        
        # Create dark memory entry
        memory_id = self._generate_memory_id(content, concept_ids)
        
        entry = SolitonMemoryEntry(
            id=memory_id,
            content=content,
            memory_type=MemoryType.TRAUMATIC,  # Dark memories often traumatic
            phase=phase,
            amplitude=1.0,  # Depth of the dip
            frequency=self._calculate_frequency(content),
            timestamp=datetime.now(timezone.utc),
            concept_ids=concept_ids,
            sources=sources or ["dark_suppression"],
            metadata={
                **(metadata or {}),
                "polarity": "dark",
                "suppression_reason": reason
            },
            polarity="dark"  # Mark as dark soliton
        )
        
        self.memory_entries[memory_id] = entry
        
        # Register oscillators for dark soliton (baseline + dip)
        lattice = get_global_lattice()
        
        # Baseline oscillator (continuous background)
        base_idx = lattice.add_oscillator(
            phase=phase,
            natural_freq=entry.frequency * 0.1,
            amplitude=1.0,  # Background amplitude
            stability=1.0
        )
        
        # Dip oscillator (π out of phase)
        dip_idx = lattice.add_oscillator(
            phase=(phase + np.pi) % (2 * np.pi),
            natural_freq=entry.frequency * 0.1,
            amplitude=entry.amplitude,  # Dip depth
            stability=1.0
        )
        
        # Strong coupling between baseline and dip
        lattice.set_coupling(base_idx, dip_idx, 1.0)
        lattice.set_coupling(dip_idx, base_idx, 1.0)
        
        # Store oscillator indices
        entry.metadata["oscillator_idx"] = dip_idx
        entry.metadata["baseline_idx"] = base_idx
        entry.baseline_oscillator_idx = base_idx
        
        logger.info(f"Stored dark soliton {memory_id} to suppress '{concept_ids[0]}': {reason}")
        
        return memory_id
    
    def recall_with_dark_suppression(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> List[SolitonMemoryEntry]:
        """Enhanced recall that respects dark soliton suppression"""
        
        # First, get all resonant memories (including dark)
        query_phase = self._calculate_query_phase(query)
        query_concepts = self._extract_concepts_nlp(query)
        
        resonant_memories = []
        dark_suppressions = set()  # Concepts suppressed by dark solitons
        
        # Phase 1: Identify dark suppressions
        for entry in self.memory_entries.values():
            if getattr(entry, 'polarity', 'bright') == 'dark':
                # Check if this dark soliton resonates with query
                resonance = self._calculate_resonance(
                    entry, query_phase, query_concepts
                )
                
                if resonance > self.resonance_threshold:
                    # This dark soliton suppresses its concepts
                    for concept in entry.concept_ids:
                        dark_suppressions.add(concept)
                    logger.debug(f"Dark soliton suppressing concepts: {entry.concept_ids}")
        
        # Phase 2: Collect bright memories not suppressed
        for entry in self.memory_entries.values():
            if getattr(entry, 'polarity', 'bright') == 'bright':
                # Skip if vaulted
                if entry.vault_status != VaultStatus.ACTIVE:
                    continue
                
                # Check if any concept is suppressed
                if any(c in dark_suppressions for c in entry.concept_ids):
                    logger.debug(f"Memory {entry.id} suppressed by dark soliton")
                    continue
                
                # Calculate resonance
                resonance = self._calculate_resonance(
                    entry, query_phase, query_concepts
                )
                
                if resonance > self.resonance_threshold:
                    resonant_memories.append((resonance, entry))
        
        # Sort by resonance
        resonant_memories.sort(key=lambda x: x[0], reverse=True)
        
        return [entry for _, entry in resonant_memories]
    
    def create_forgetting_memory(
        self,
        concept_to_forget: str,
        strength: float = 1.0
    ) -> str:
        """Create a dark soliton to forget/suppress a concept"""
        
        return self.store_dark_memory(
            content=f"[SUPPRESSION: Forget {concept_to_forget}]",
            concept_ids=[concept_to_forget],
            reason="intentional_forgetting",
            metadata={
                "forget_strength": strength,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        )
    
    def cancel_memories_by_concept(
        self,
        concept: str,
        reason: str = "contradiction"
    ) -> int:
        """Use dark soliton to cancel all memories of a concept"""
        
        # Find all bright memories with this concept
        affected_count = 0
        for entry in self.memory_entries.values():
            if (getattr(entry, 'polarity', 'bright') == 'bright' and 
                concept in entry.concept_ids):
                affected_count += 1
        
        if affected_count > 0:
            # Create strong dark soliton
            self.store_dark_memory(
                content=f"[CANCELLATION: {concept}]",
                concept_ids=[concept],
                reason=reason,
                metadata={
                    "affected_memories": affected_count,
                    "cancellation_type": "total"
                }
            )
            
            logger.info(f"Created dark soliton to cancel {affected_count} memories "
                       f"of concept '{concept}'")
        
        return affected_count


# Patch for oscillator lattice to handle dark solitons
class DarkSolitonLatticeEnhancements:
    """Enhancements for oscillator lattice to support dark solitons"""
    
    def add_dark_soliton_pair(
        self,
        phase: float,
        frequency: float,
        dip_amplitude: float = 1.0,
        baseline_amplitude: float = 1.0
    ) -> Tuple[int, int]:
        """
        Add a dark soliton as a baseline-dip oscillator pair
        Returns (baseline_idx, dip_idx)
        """
        # Add baseline oscillator
        base_idx = self.add_oscillator(
            phase=phase,
            natural_freq=frequency,
            amplitude=baseline_amplitude,
            stability=1.0
        )
        
        # Add dip oscillator (π out of phase)
        dip_idx = self.add_oscillator(
            phase=(phase + np.pi) % (2 * np.pi),
            natural_freq=frequency,
            amplitude=dip_amplitude,
            stability=1.0
        )
        
        # Strong coupling between them
        self.set_coupling(base_idx, dip_idx, 1.0)
        self.set_coupling(dip_idx, base_idx, 1.0)
        
        # Mark as dark soliton pair in metadata
        self.oscillators[base_idx]['soliton_type'] = 'dark_baseline'
        self.oscillators[dip_idx]['soliton_type'] = 'dark_dip'
        self.oscillators[base_idx]['pair_idx'] = dip_idx
        self.oscillators[dip_idx]['pair_idx'] = base_idx
        
        return base_idx, dip_idx
    
    def evaluate_dark_soliton_field(
        self,
        base_idx: int,
        dip_idx: int,
        positions: np.ndarray
    ) -> np.ndarray:
        """Evaluate the field of a dark soliton at given positions"""
        
        if base_idx >= len(self.oscillators) or dip_idx >= len(self.oscillators):
            return np.zeros_like(positions)
        
        base_osc = self.oscillators[base_idx]
        dip_osc = self.oscillators[dip_idx]
        
        # Baseline field
        baseline = base_osc['amplitude'] * np.ones_like(positions)
        
        # Dip profile (tanh shape for dark soliton)
        center = base_osc.get('position', 0.0)
        width = 1.0  # Characteristic width
        
        dip_profile = dip_osc['amplitude'] * np.tanh((positions - center) / width)
        
        # Dark soliton field = baseline with dip
        field = baseline * np.sqrt(1 - dip_profile**2)
        
        # Phase jump at center
        phase = base_osc['phase'] + np.pi * (1 + np.tanh((positions - center) / width)) / 2
        
        return field * np.exp(1j * phase)


# Integration patch to add dark soliton support to existing classes
def patch_dark_soliton_support():
    """Apply dark soliton patches to existing classes"""
    
    # Patch EnhancedSolitonMemory
    from python.core.soliton_memory_integration import EnhancedSolitonMemory
    
    # Add dark soliton methods
    EnhancedSolitonMemory.store_dark_memory = DarkSolitonMemoryEnhancements.store_dark_memory
    EnhancedSolitonMemory.recall_with_dark_suppression = DarkSolitonMemoryEnhancements.recall_with_dark_suppression
    EnhancedSolitonMemory.create_forgetting_memory = DarkSolitonMemoryEnhancements.create_forgetting_memory
    EnhancedSolitonMemory.cancel_memories_by_concept = DarkSolitonMemoryEnhancements.cancel_memories_by_concept
    
    # Patch OscillatorLattice
    from python.core.oscillator_lattice import OscillatorLattice
    
    OscillatorLattice.add_dark_soliton_pair = DarkSolitonLatticeEnhancements.add_dark_soliton_pair
    OscillatorLattice.evaluate_dark_soliton_field = DarkSolitonLatticeEnhancements.evaluate_dark_soliton_field
    
    logger.info("Dark soliton support patches applied")


# Example usage and testing
if __name__ == "__main__":
    # Apply patches
    patch_dark_soliton_support()
    
    # Test dark soliton functionality
    from python.core.soliton_memory_integration import EnhancedSolitonMemory
    from python.core.oscillator_lattice import get_global_lattice
    
    # Create memory system
    memory_system = EnhancedSolitonMemory(lattice_size=1000)
    
    # Store a bright memory
    bright_id = memory_system.store_enhanced_memory(
        content="The sky is blue",
        concept_ids=["sky", "blue"],
        memory_type=MemoryType.SEMANTIC
    )
    
    # Store a dark memory to suppress it
    dark_id = memory_system.store_dark_memory(
        content="[SUPPRESS: sky color]",
        concept_ids=["sky"],
        reason="testing dark soliton"
    )
    
    # Test recall - should not return the bright memory
    results = memory_system.recall_with_dark_suppression("What color is the sky?")
    
    print(f"Stored bright memory: {bright_id}")
    print(f"Stored dark memory: {dark_id}")
    print(f"Recall results: {len(results)} memories")
    
    # Test forgetting
    memory_system.create_forgetting_memory("blue", strength=1.0)
    
    # Test cancellation
    affected = memory_system.cancel_memories_by_concept("sky", reason="test cancellation")
    print(f"Cancelled {affected} memories about 'sky'")
