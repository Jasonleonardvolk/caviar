#!/usr/bin/env python3
"""
Complete fixes for soliton_memory_integration.py
Addresses 2 issues: memory fusion cleanup and fission implementation
"""

from collections import defaultdict
import numpy as np
from datetime import datetime, timezone
import logging

# Import get_global_lattice with fallback
try:
    from python.core.oscillator_lattice import get_global_lattice
except ImportError:
    try:
        from oscillator_lattice import get_global_lattice
    except ImportError:
        from core.oscillator_lattice import get_global_lattice

# Import SolitonMemoryEntry and VaultStatus
try:
    from python.core.soliton_memory_integration import SolitonMemoryEntry, VaultStatus
except ImportError:
    from soliton_memory_integration import SolitonMemoryEntry, VaultStatus

logger = logging.getLogger(__name__)

class EnhancedSolitonMemory:
    
    # FIX 1: Complete memory fusion with proper oscillator cleanup
    def _perform_memory_fusion(self) -> int:
        """Fuse similar memories and properly clean up ALL oscillators"""
        fused = 0
        concept_groups = defaultdict(list)
        
        # Group by primary concept
        for mem_id, entry in self.memory_entries.items():
            if entry.concept_ids:
                concept_groups[entry.concept_ids[0]].append((mem_id, entry))
        
        # Get global lattice for oscillator management
        lattice = get_global_lattice()
        
        # Track oscillators to remove
        oscillators_to_remove = set()
        
        # Fuse duplicates
        for concept, memories in concept_groups.items():
            if len(memories) > 1:
                # Sort by amplitude (strongest first)
                memories.sort(key=lambda x: x[1].amplitude, reverse=True)
                main_id, main_entry = memories[0]
                
                # Check similarity threshold for fusion
                fusion_candidates = []
                for other_id, other_entry in memories[1:]:
                    if self._should_fuse(main_entry, other_entry):
                        fusion_candidates.append((other_id, other_entry))
                
                # Perform fusion
                for other_id, other_entry in fusion_candidates:
                    # Combine properties
                    main_entry.amplitude = min(2.0, 
                        np.sqrt(main_entry.amplitude**2 + other_entry.amplitude**2))
                    main_entry.heat = max(main_entry.heat, other_entry.heat)
                    main_entry.access_count += other_entry.access_count
                    
                    # Merge content intelligently
                    if len(main_entry.content) < 500:  # Don't make it too long
                        main_entry.content += f" [Fused: {other_entry.content[:100]}...]"
                    
                    # Collect oscillators to remove
                    if "oscillator_idx" in other_entry.metadata:
                        oscillators_to_remove.add(other_entry.metadata["oscillator_idx"])
                    
                    # For dark solitons, also remove baseline
                    if other_entry.polarity == "dark" and "baseline_idx" in other_entry.metadata:
                        oscillators_to_remove.add(other_entry.metadata["baseline_idx"])
                    
                    # Remove the memory entry
                    self.memory_entries.pop(other_id, None)
                    fused += 1
                    
                    logger.info(f"Fused memory {other_id} into {main_id}")
        
        # Clean up oscillators in batch (more efficient)
        if oscillators_to_remove:
            # Sort in descending order to avoid index shifting issues
            for idx in sorted(oscillators_to_remove, reverse=True):
                if idx < len(lattice.oscillators):
                    # Don't actually remove, just mark inactive
                    lattice.oscillators[idx]["active"] = False
                    lattice.oscillators[idx]["amplitude"] = 0.0
            
            # Actually purge inactive oscillators and rebuild the Laplacian
            if hasattr(lattice, 'purge_inactive_oscillators'):
                lattice.purge_inactive_oscillators()
            else:
                # Manual purging if method doesn't exist
                active_oscillators = []
                active_indices = []
                for i, osc in enumerate(lattice.oscillators):
                    if osc.get('active', True) and osc.get('amplitude', 0) > 1e-10:
                        active_oscillators.append(osc)
                        active_indices.append(i)
                
                # Update oscillator list
                lattice.oscillators = active_oscillators
                
                # Rebuild coupling matrix
                if hasattr(lattice, 'K') and lattice.K is not None:
                    old_K = lattice.K
                    new_size = len(active_oscillators)
                    lattice.K = np.zeros((new_size, new_size))
                    
                    # Copy over active couplings
                    for new_i, old_i in enumerate(active_indices):
                        for new_j, old_j in enumerate(active_indices):
                            if old_i < old_K.shape[0] and old_j < old_K.shape[1]:
                                lattice.K[new_i, new_j] = old_K[old_i, old_j]
            
            # Alternative: if purge method doesn't exist, rebuild coupling matrix
            # This ensures the dense matrix stays small
            if hasattr(lattice, 'rebuild_laplacian'):
                lattice.rebuild_laplacian()
            
            logger.info(f"Purged {len(oscillators_to_remove)} oscillators and rebuilt lattice")
        
        return fused
    
    # FIX 2: Implement memory fission for complex memories
    def _perform_memory_fission(self) -> int:
        """Split complex memories into smaller, more manageable ones"""
        split = 0
        lattice = get_global_lattice()
        
        # Find memories that should be split
        memories_to_split = []
        for mem_id, entry in self.memory_entries.items():
            if self._should_split(entry):
                memories_to_split.append((mem_id, entry))
        
        logger.info(f"Found {len(memories_to_split)} memories to split")
        
        for mem_id, entry in memories_to_split:
            # Determine split strategy
            if len(entry.content) > 1000:
                # Content-based split
                split_memories = self._split_by_content(entry)
            elif entry.amplitude > 1.5:
                # Amplitude-based split
                split_memories = self._split_by_amplitude(entry)
            elif len(entry.concept_ids) > 3:
                # Concept-based split
                split_memories = self._split_by_concepts(entry)
            else:
                continue
            
            # Store split memories
            for new_entry in split_memories:
                # Add oscillator for new memory
                osc_idx = lattice.add_oscillator(
                    phase=new_entry.phase,
                    natural_freq=new_entry.frequency * 0.1,
                    amplitude=new_entry.amplitude,
                    stability=0.8
                )
                new_entry.metadata["oscillator_idx"] = osc_idx
                new_entry.metadata["split_from"] = mem_id
                new_entry.metadata["split_time"] = datetime.now(timezone.utc).isoformat()
                
                # Store new entry
                self.memory_entries[new_entry.id] = new_entry
                
                logger.debug(f"Created split memory {new_entry.id} from {mem_id}")
            
            # Remove original memory and its oscillator
            if "oscillator_idx" in entry.metadata:
                idx = entry.metadata["oscillator_idx"]
                if idx < len(lattice.oscillators):
                    lattice.oscillators[idx]["active"] = False
            
            # Remove original entry
            self.memory_entries.pop(mem_id, None)
            split += 1
        
        logger.info(f"Split {split} memories into {split * 2} smaller ones")
        return split
    
    def _should_split(self, memory: SolitonMemoryEntry) -> bool:
        """Determine if a memory should be split"""
        # Multiple criteria for splitting
        return (
            memory.amplitude > 1.5 or                    # Too strong
            len(memory.content) > 1000 or                # Too long
            len(memory.concept_ids) > 3 or               # Too many concepts
            memory.heat > 0.9 or                         # Too hot
            memory.access_count > 100                    # Accessed too much
        )
    
    def _split_by_content(self, entry: SolitonMemoryEntry) -> list:
        """Split memory based on content length"""
        content = entry.content
        mid = len(content) // 2
        
        # Find a good split point (end of sentence/word)
        for offset in range(min(100, mid)):
            if mid + offset < len(content) and content[mid + offset] in '.!?\n':
                mid = mid + offset + 1
                break
            elif mid - offset > 0 and content[mid - offset] in '.!?\n':
                mid = mid - offset + 1
                break
        
        # Create two new entries
        entry1 = SolitonMemoryEntry(
            id=self._generate_memory_id(content[:mid], entry.concept_ids),
            content=content[:mid].strip(),
            memory_type=entry.memory_type,
            phase=entry.phase,
            amplitude=entry.amplitude * 0.6,
            frequency=entry.frequency,
            timestamp=datetime.now(timezone.utc),
            concept_ids=entry.concept_ids,
            sources=entry.sources,
            metadata={"split_type": "content"}
        )
        
        entry2 = SolitonMemoryEntry(
            id=self._generate_memory_id(content[mid:], entry.concept_ids),
            content=content[mid:].strip(),
            memory_type=entry.memory_type,
            phase=(entry.phase + np.pi/3) % (2 * np.pi),  # Slight phase shift
            amplitude=entry.amplitude * 0.6,
            frequency=entry.frequency,
            timestamp=datetime.now(timezone.utc),
            concept_ids=entry.concept_ids,
            sources=entry.sources,
            metadata={"split_type": "content"}
        )
        
        return [entry1, entry2]
    
    def _split_by_amplitude(self, entry: SolitonMemoryEntry) -> list:
        """Split memory based on high amplitude"""
        # Create two memories with reduced amplitude
        entry1 = SolitonMemoryEntry(
            id=self._generate_memory_id(entry.content + "_primary", entry.concept_ids),
            content=entry.content,
            memory_type=entry.memory_type,
            phase=entry.phase,
            amplitude=entry.amplitude * 0.5,
            frequency=entry.frequency,
            timestamp=datetime.now(timezone.utc),
            concept_ids=entry.concept_ids[:2] if len(entry.concept_ids) > 1 else entry.concept_ids,
            sources=entry.sources,
            metadata={"split_type": "amplitude", "split_role": "primary"}
        )
        
        entry2 = SolitonMemoryEntry(
            id=self._generate_memory_id(entry.content + "_secondary", entry.concept_ids),
            content=entry.content,
            memory_type=entry.memory_type,
            phase=(entry.phase + np.pi/2) % (2 * np.pi),  # 90° phase shift
            amplitude=entry.amplitude * 0.5,
            frequency=entry.frequency * 0.9,  # Slightly different frequency
            timestamp=datetime.now(timezone.utc),
            concept_ids=entry.concept_ids[1:] if len(entry.concept_ids) > 1 else entry.concept_ids,
            sources=entry.sources,
            metadata={"split_type": "amplitude", "split_role": "secondary"}
        )
        
        return [entry1, entry2]
    
    def _split_by_concepts(self, entry: SolitonMemoryEntry) -> list:
        """Split memory based on multiple concepts"""
        # Divide concepts between two memories
        mid = len(entry.concept_ids) // 2
        concepts1 = entry.concept_ids[:mid]
        concepts2 = entry.concept_ids[mid:]
        
        entry1 = SolitonMemoryEntry(
            id=self._generate_memory_id(entry.content, concepts1),
            content=entry.content,
            memory_type=entry.memory_type,
            phase=self._calculate_concept_phase(concepts1),
            amplitude=entry.amplitude * 0.7,
            frequency=entry.frequency,
            timestamp=datetime.now(timezone.utc),
            concept_ids=concepts1,
            sources=entry.sources,
            metadata={"split_type": "concepts", "original_concepts": entry.concept_ids}
        )
        
        entry2 = SolitonMemoryEntry(
            id=self._generate_memory_id(entry.content, concepts2),
            content=entry.content,
            memory_type=entry.memory_type,
            phase=self._calculate_concept_phase(concepts2),
            amplitude=entry.amplitude * 0.7,
            frequency=entry.frequency,
            timestamp=datetime.now(timezone.utc),
            concept_ids=concepts2,
            sources=entry.sources,
            metadata={"split_type": "concepts", "original_concepts": entry.concept_ids}
        )
        
        return [entry1, entry2]
    
    def _should_fuse(self, mem1: SolitonMemoryEntry, mem2: SolitonMemoryEntry) -> bool:
        """Enhanced fusion criteria"""
        # Don't fuse if already split
        if "split_from" in mem1.metadata or "split_from" in mem2.metadata:
            # Check if they were split recently
            if "split_time" in mem1.metadata:
                split_time = datetime.fromisoformat(mem1.metadata["split_time"].replace('Z', '+00:00'))
                if (datetime.now(timezone.utc) - split_time).total_seconds() < 3600:  # 1 hour
                    return False
        
        # Same concept and similar properties
        if mem1.concept_ids == mem2.concept_ids:
            # Check content similarity (simple overlap check)
            content1_words = set(mem1.content.lower().split())
            content2_words = set(mem2.content.lower().split())
            overlap = len(content1_words & content2_words)
            total = len(content1_words | content2_words)
            
            if total > 0 and overlap / total > 0.7:  # 70% similarity
                return True
        
        return False
