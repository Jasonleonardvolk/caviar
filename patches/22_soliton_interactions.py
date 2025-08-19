#!/usr/bin/env python3
"""
Soliton Fission, Fusion, and Collision Logic
Implements memory splitting, merging, and dark/bright collision handling
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timezone
from collections import defaultdict
import asyncio

from python.core.soliton_memory_integration import (
    EnhancedSolitonMemory, SolitonMemoryEntry, MemoryType, VaultStatus
)
from python.core.oscillator_lattice import get_global_lattice

logger = logging.getLogger(__name__)


class SolitonInteractionEngine:
    """Handles soliton fission, fusion, and collision dynamics"""
    
    def __init__(self, memory_system: EnhancedSolitonMemory):
        self.memory_system = memory_system
        self.fusion_threshold = 0.8  # Similarity threshold for fusion
        self.fission_complexity_threshold = 3  # Number of concepts before splitting
        self.fission_amplitude_threshold = 1.5  # Amplitude before splitting
        self.collision_resolution_threshold = 0.8  # Strength ratio for collision winner
        
    async def run_consolidation_cycle(self) -> Dict[str, int]:
        """Run a complete consolidation cycle with fission, fusion, and collision handling"""
        logger.info("=== Starting Soliton Consolidation Cycle ===")
        
        results = {
            'fused': 0,
            'split': 0,
            'collisions_resolved': 0,
            'memories_suppressed': 0
        }
        
        # Phase 1: Detect and handle collisions (bright vs dark)
        collision_results = self._handle_collisions()
        results['collisions_resolved'] = collision_results['resolved']
        results['memories_suppressed'] = collision_results['suppressed']
        
        # Phase 2: Fuse similar memories
        fusion_results = self._perform_fusion()
        results['fused'] = fusion_results['fused']
        
        # Phase 3: Split complex memories
        fission_results = self._perform_fission()
        results['split'] = fission_results['split']
        
        logger.info(f"Consolidation complete: {results}")
        return results
    
    def _handle_collisions(self) -> Dict[str, int]:
        """Handle bright vs dark soliton collisions"""
        results = {'resolved': 0, 'suppressed': 0}
        
        # Group memories by concept
        concept_groups = defaultdict(list)
        for mem_id, entry in self.memory_system.memory_entries.items():
            if entry.concept_ids:
                primary_concept = entry.concept_ids[0]
                concept_groups[primary_concept].append((mem_id, entry))
        
        # Check each concept group for bright/dark pairs
        for concept, memories in concept_groups.items():
            bright_memories = [(mid, mem) for mid, mem in memories 
                             if getattr(mem, 'polarity', 'bright') == 'bright']
            dark_memories = [(mid, mem) for mid, mem in memories 
                            if getattr(mem, 'polarity', 'bright') == 'dark']
            
            if bright_memories and dark_memories:
                # We have a collision - resolve it
                logger.info(f"Collision detected for concept '{concept}': "
                          f"{len(bright_memories)} bright vs {len(dark_memories)} dark")
                
                # Calculate total strengths
                bright_strength = sum(mem.amplitude for _, mem in bright_memories)
                dark_strength = sum(mem.amplitude for _, mem in dark_memories)
                
                if dark_strength >= bright_strength * self.collision_resolution_threshold:
                    # Dark wins - suppress all bright memories
                    for mem_id, memory in bright_memories:
                        memory.vault_status = VaultStatus.QUARANTINE
                        logger.debug(f"Suppressed bright memory {mem_id}")
                        results['suppressed'] += 1
                    
                    # Dark memory has done its job, can be removed
                    for mem_id, _ in dark_memories:
                        del self.memory_system.memory_entries[mem_id]
                        self._remove_oscillator(mem_id)
                else:
                    # Bright wins - remove ineffective dark memories
                    for mem_id, _ in dark_memories:
                        del self.memory_system.memory_entries[mem_id]
                        self._remove_oscillator(mem_id)
                        logger.debug(f"Removed weak dark memory {mem_id}")
                
                results['resolved'] += 1
        
        return results
    
    def _perform_fusion(self) -> Dict[str, int]:
        """Fuse similar memories together"""
        results = {'fused': 0}
        lattice = get_global_lattice()
        
        # Group by concept
        concept_groups = defaultdict(list)
        for mem_id, entry in list(self.memory_system.memory_entries.items()):
            if entry.concept_ids and getattr(entry, 'polarity', 'bright') == 'bright':
                concept_groups[entry.concept_ids[0]].append((mem_id, entry))
        
        # Check each group for fusion candidates
        for concept, memories in concept_groups.items():
            if len(memories) < 2:
                continue
            
            # Sort by amplitude (strongest first)
            memories.sort(key=lambda x: x[1].amplitude, reverse=True)
            
            i = 0
            while i < len(memories) - 1:
                main_id, main_mem = memories[i]
                
                # Check subsequent memories for fusion
                j = i + 1
                while j < len(memories):
                    other_id, other_mem = memories[j]
                    
                    if self._should_fuse(main_mem, other_mem):
                        # Perform fusion
                        self._fuse_memories(main_mem, other_mem, main_id, other_id)
                        
                        # Remove fused memory
                        del self.memory_system.memory_entries[other_id]
                        memories.pop(j)
                        results['fused'] += 1
                        
                        logger.info(f"Fused memory {other_id} into {main_id}")
                    else:
                        j += 1
                
                i += 1
        
        return results
    
    def _should_fuse(self, mem1: SolitonMemoryEntry, mem2: SolitonMemoryEntry) -> bool:
        """Determine if two memories should be fused"""
        # Don't fuse if recently split
        if ('split_from' in mem1.metadata or 'split_from' in mem2.metadata):
            return False
        
        # Check phase similarity
        phase_diff = abs(mem1.phase - mem2.phase)
        if phase_diff > 0.1:  # Different phases, don't fuse
            return False
        
        # Check content similarity (simple word overlap)
        if hasattr(mem1, 'content') and hasattr(mem2, 'content'):
            words1 = set(mem1.content.lower().split())
            words2 = set(mem2.content.lower().split())
            
            if words1 and words2:
                overlap = len(words1 & words2)
                union = len(words1 | words2)
                similarity = overlap / union
                
                return similarity > self.fusion_threshold
        
        return False
    
    def _fuse_memories(self, main: SolitonMemoryEntry, other: SolitonMemoryEntry,
                      main_id: str, other_id: str):
        """Fuse two memories together"""
        # Combine amplitudes (energy conservation)
        main.amplitude = np.sqrt(main.amplitude**2 + other.amplitude**2)
        main.amplitude = min(main.amplitude, 2.0)  # Cap at maximum
        
        # Combine heat and access
        main.heat = max(main.heat, other.heat)
        main.access_count += other.access_count
        
        # Merge sources
        if hasattr(other, 'sources'):
            main.sources = list(set(main.sources + other.sources))
        
        # Update metadata
        main.metadata['fusion_count'] = main.metadata.get('fusion_count', 0) + 1
        main.metadata['fused_from'] = main.metadata.get('fused_from', []) + [other_id]
        
        # Remove other's oscillator
        self._remove_oscillator(other_id)
    
    def _perform_fission(self) -> Dict[str, int]:
        """Split complex memories into simpler ones"""
        results = {'split': 0}
        lattice = get_global_lattice()
        
        memories_to_split = []
        
        # Find candidates for splitting
        for mem_id, entry in list(self.memory_system.memory_entries.items()):
            if self._should_split(entry):
                memories_to_split.append((mem_id, entry))
        
        # Perform splits
        for mem_id, entry in memories_to_split:
            new_memories = self._split_memory(entry)
            
            if len(new_memories) > 1:
                # Remove original
                del self.memory_system.memory_entries[mem_id]
                self._remove_oscillator(mem_id)
                
                # Add new memories
                for new_entry in new_memories:
                    # Store using memory system
                    new_id = self.memory_system.store_enhanced_memory(
                        content=new_entry.content,
                        concept_ids=new_entry.concept_ids,
                        memory_type=new_entry.memory_type,
                        sources=new_entry.sources,
                        metadata={
                            'split_from': mem_id,
                            'split_time': datetime.now(timezone.utc).isoformat(),
                            **new_entry.metadata
                        }
                    )
                    
                    # Copy over important properties
                    if new_id in self.memory_system.memory_entries:
                        stored = self.memory_system.memory_entries[new_id]
                        stored.amplitude = new_entry.amplitude
                        stored.heat = new_entry.heat * 0.8  # Reduce heat after split
                
                results['split'] += 1
                logger.info(f"Split memory {mem_id} into {len(new_memories)} parts")
        
        return results
    
    def _should_split(self, memory: SolitonMemoryEntry) -> bool:
        """Determine if a memory should be split"""
        # Multiple criteria
        if len(memory.concept_ids) >= self.fission_complexity_threshold:
            return True
        
        if memory.amplitude > self.fission_amplitude_threshold:
            return True
        
        if hasattr(memory, 'content') and len(memory.content) > 1000:
            return True
        
        # High frequency can indicate complexity
        if memory.frequency > 2.0:
            return True
        
        return False
    
    def _split_memory(self, memory: SolitonMemoryEntry) -> List[SolitonMemoryEntry]:
        """Split a memory into multiple parts"""
        new_memories = []
        
        # Strategy 1: Split by concepts
        if len(memory.concept_ids) > 1:
            # Create one memory per concept
            for concept in memory.concept_ids:
                new_entry = SolitonMemoryEntry(
                    id=f"{memory.id}_split_{concept}",
                    content=memory.content,
                    memory_type=memory.memory_type,
                    phase=self.memory_system._calculate_concept_phase([concept]),
                    amplitude=memory.amplitude / np.sqrt(len(memory.concept_ids)),
                    frequency=memory.frequency,
                    timestamp=datetime.now(timezone.utc),
                    concept_ids=[concept],
                    sources=memory.sources,
                    metadata={'split_type': 'concept'}
                )
                new_memories.append(new_entry)
        
        # Strategy 2: Split by amplitude (energy distribution)
        elif memory.amplitude > self.fission_amplitude_threshold:
            # Split into two equal-energy parts
            for i in range(2):
                new_entry = SolitonMemoryEntry(
                    id=f"{memory.id}_split_{i}",
                    content=memory.content,
                    memory_type=memory.memory_type,
                    phase=memory.phase + (i * np.pi/4),  # Slight phase shift
                    amplitude=memory.amplitude / np.sqrt(2),
                    frequency=memory.frequency,
                    timestamp=datetime.now(timezone.utc),
                    concept_ids=memory.concept_ids,
                    sources=memory.sources,
                    metadata={'split_type': 'amplitude', 'part': i}
                )
                new_memories.append(new_entry)
        
        # Strategy 3: Content-based split (if content is long)
        elif hasattr(memory, 'content') and len(memory.content) > 1000:
            # Simple split at sentence boundaries
            sentences = memory.content.split('. ')
            mid = len(sentences) // 2
            
            part1 = '. '.join(sentences[:mid]) + '.'
            part2 = '. '.join(sentences[mid:]) + '.'
            
            for i, content in enumerate([part1, part2]):
                new_entry = SolitonMemoryEntry(
                    id=f"{memory.id}_split_{i}",
                    content=content,
                    memory_type=memory.memory_type,
                    phase=memory.phase + (i * np.pi/6),
                    amplitude=memory.amplitude * 0.7,
                    frequency=memory.frequency,
                    timestamp=datetime.now(timezone.utc),
                    concept_ids=memory.concept_ids,
                    sources=memory.sources,
                    metadata={'split_type': 'content', 'part': i}
                )
                new_memories.append(new_entry)
        
        # If no split performed, return original
        return new_memories if new_memories else [memory]
    
    def _remove_oscillator(self, memory_id: str):
        """Remove oscillator associated with a memory"""
        lattice = get_global_lattice()
        
        # Find the oscillator index from memory metadata
        for mem_id, entry in self.memory_system.memory_entries.items():
            if mem_id == memory_id and 'oscillator_idx' in entry.metadata:
                idx = entry.metadata['oscillator_idx']
                if idx < len(lattice.oscillators):
                    lattice.oscillators[idx]['active'] = False
                    lattice.oscillators[idx]['amplitude'] = 0.0
                    
                # For dark solitons, also deactivate baseline
                if 'baseline_idx' in entry.metadata:
                    base_idx = entry.metadata['baseline_idx']
                    if base_idx < len(lattice.oscillators):
                        lattice.oscillators[base_idx]['active'] = False


class SolitonVotingSystem:
    """Implements soliton voting for collective memory decisions"""
    
    def __init__(self, memory_system: EnhancedSolitonMemory):
        self.memory_system = memory_system
        
    def compute_concept_votes(self) -> Dict[str, float]:
        """Compute net votes for each concept (bright - dark)"""
        concept_votes = defaultdict(float)
        
        for entry in self.memory_system.memory_entries.values():
            if not entry.concept_ids:
                continue
                
            primary_concept = entry.concept_ids[0]
            vote_weight = entry.amplitude * entry.stability
            
            # Bright memories vote positive, dark vote negative
            if getattr(entry, 'polarity', 'bright') == 'bright':
                concept_votes[primary_concept] += vote_weight
            else:
                concept_votes[primary_concept] -= vote_weight
        
        return dict(concept_votes)
    
    def apply_voting_decisions(self, votes: Dict[str, float]) -> Dict[str, int]:
        """Apply voting results to suppress or reinforce memories"""
        results = {'suppressed': 0, 'reinforced': 0, 'weakened': 0}
        
        for concept, net_vote in votes.items():
            if net_vote <= 0:
                # Negative vote - suppress all memories of this concept
                for entry in self.memory_system.memory_entries.values():
                    if entry.concept_ids and entry.concept_ids[0] == concept:
                        entry.vault_status = VaultStatus.QUARANTINE
                        results['suppressed'] += 1
                        logger.info(f"Suppressed concept '{concept}' (vote: {net_vote:.2f})")
            
            elif net_vote < 0.5:
                # Weak positive - weaken memories
                for entry in self.memory_system.memory_entries.values():
                    if entry.concept_ids and entry.concept_ids[0] == concept:
                        entry.amplitude *= 0.8
                        entry.stability *= 0.9
                        results['weakened'] += 1
            
            else:
                # Strong positive - reinforce memories
                for entry in self.memory_system.memory_entries.values():
                    if entry.concept_ids and entry.concept_ids[0] == concept:
                        entry.stability = min(1.0, entry.stability * 1.1)
                        results['reinforced'] += 1
        
        return results
