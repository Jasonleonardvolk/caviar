# python/core/memory_crystallization.py

import asyncio
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import numpy as np

from .soliton_memory_integration import EnhancedSolitonMemory, SolitonMemoryEntry, VaultStatus
from .hot_swap_laplacian import HotSwappableLaplacian
from .topology_policy import get_topology_policy
from .oscillator_lattice import get_global_lattice

logger = logging.getLogger(__name__)

class MemoryCrystallizer:
    """
    Handles nightly memory crystallization process.
    Reorganizes memories based on heat/importance.
    """
    
    def __init__(self, 
                 memory_system: EnhancedSolitonMemory,
                 hot_swap: HotSwappableLaplacian):
        self.memory = memory_system
        self.hot_swap = hot_swap
        self.policy = get_topology_policy(hot_swap)
        
        # Crystallization parameters
        self.hot_threshold = 0.7
        self.warm_threshold = 0.4
        self.cold_threshold = 0.1
        self.decay_threshold = 0.05
        
        # Migration settings
        self.migration_batch_size = 10
        self.migration_delay = 0.1  # seconds between batches
        
    async def crystallize(self) -> Dict[str, Any]:
        """
        Main crystallization process.
        Runs during nightly maintenance.
        """
        logger.info("=== Starting Memory Crystallization ===")
        start_time = datetime.now()
        
        report = {
            "start_time": start_time.isoformat(),
            "initial_count": len(self.memory.memory_entries),
            "migrated": 0,
            "decayed": 0,
            "fused": 0,
            "split": 0,
            "topology_switches": 0
        }
        
        try:
            # 1. Enter consolidation mode
            restore_topology = await self.policy.nightly_consolidation_mode()
            report["topology_switches"] += 1
            
            # 2. Let system settle
            await asyncio.sleep(2.0)
            
            # 3. Categorize memories by heat
            hot, warm, cold = self._categorize_memories()
            
            logger.info(f"Memory distribution: {len(hot)} hot, {len(warm)} warm, {len(cold)} cold")
            
            # 4. Migrate hot memories to stable positions
            report["migrated"] = await self._migrate_hot_memories(hot)
            
            # 5. Allow cold memories to decay
            report["decayed"] = await self._decay_cold_memories(cold)
            
            # 6. Perform fusion on similar memories
            report["fused"] = await self._fuse_similar_memories()
            
            # 7. Perform fission on complex memories
            report["split"] = await self._split_complex_memories()
            
            # 8. Restore normal topology
            await restore_topology()
            report["topology_switches"] += 1
            
            # 9. Final cleanup
            self._cleanup_empty_oscillators()
            
        except Exception as e:
            logger.error(f"Crystallization error: {e}")
            report["error"] = str(e)
            
        finally:
            report["end_time"] = datetime.now().isoformat()
            report["duration"] = (datetime.now() - start_time).total_seconds()
            report["final_count"] = len(self.memory.memory_entries)
            
            logger.info(f"Crystallization complete: {report}")
            return report
            
    def _categorize_memories(self) -> Tuple[List[SolitonMemoryEntry], List[SolitonMemoryEntry], List[SolitonMemoryEntry]]:
        """Categorize memories by heat level"""
        hot = []
        warm = []
        cold = []
        
        for entry in self.memory.memory_entries.values():
            if entry.heat > self.hot_threshold:
                hot.append(entry)
            elif entry.heat > self.cold_threshold:
                warm.append(entry)
            else:
                cold.append(entry)
                
        return hot, warm, cold
        
    async def _migrate_hot_memories(self, hot_memories: List[SolitonMemoryEntry]) -> int:
        """Migrate hot memories to stable lattice positions"""
        migrated = 0
        
        # In Kagome topology, we want hot memories in the center triangles
        # This is simplified - real implementation would use actual topology info
        
        for i in range(0, len(hot_memories), self.migration_batch_size):
            batch = hot_memories[i:i + self.migration_batch_size]
            
            for memory in batch:
                if self._needs_migration(memory):
                    self._perform_migration(memory)
                    migrated += 1
                    
            # Small delay between batches
            await asyncio.sleep(self.migration_delay)
            
        logger.info(f"Migrated {migrated} hot memories to stable positions")
        return migrated
        
    def _needs_migration(self, memory: SolitonMemoryEntry) -> bool:
        """Check if memory needs migration"""
        # Simplified check - in reality would check lattice position
        return memory.stability < 0.8 and memory.heat > self.hot_threshold
        
    def _perform_migration(self, memory: SolitonMemoryEntry):
        """Migrate memory to more stable position"""
        # Update stability as if moved to better position
        memory.stability = min(0.95, memory.stability + 0.15)
        
        # In real implementation, would update oscillator couplings
        if "oscillator_idx" in memory.metadata:
            # Placeholder for actual lattice repositioning
            pass
            
    async def _decay_cold_memories(self, cold_memories: List[SolitonMemoryEntry]) -> int:
        """Allow cold memories to naturally decay"""
        decayed = 0
        
        for memory in cold_memories:
            # Reduce amplitude
            memory.amplitude *= 0.9
            
            # If below threshold, mark for removal
            if memory.amplitude < self.decay_threshold:
                memory.vault_status = VaultStatus.QUARANTINE
                decayed += 1
                
                # Remove oscillator
                if "oscillator_idx" in memory.metadata:
                    idx = memory.metadata["oscillator_idx"]
                    lattice = get_global_lattice()
                    # Mark as inactive rather than deleting
                    if idx < len(lattice.oscillators):
                        lattice.oscillators[idx]["active"] = False
                        
        logger.info(f"Decayed {decayed} cold memories")
        return decayed
        
    async def _fuse_similar_memories(self) -> int:
        """Fuse similar/duplicate memories"""
        fused = self.memory._perform_memory_fusion()
        logger.info(f"Fused {fused} similar memories")
        return fused
        
    async def _split_complex_memories(self) -> int:
        """Split overly complex memories"""
        split = 0
        
        for entry_id, entry in list(self.memory.memory_entries.items()):
            if self._should_split(entry):
                new_memories = self._perform_split(entry)
                if new_memories:
                    # Remove original
                    self.memory.memory_entries.pop(entry_id, None)
                    
                    # Add new memories
                    for new_entry in new_memories:
                        self.memory.memory_entries[new_entry.id] = new_entry
                        
                    split += 1
                    
        logger.info(f"Split {split} complex memories")
        return split
        
    def _should_split(self, memory: SolitonMemoryEntry) -> bool:
        """Determine if memory should be split"""
        # Multiple concepts and high complexity
        return (len(memory.concept_ids) > 2 and 
                memory.frequency > 0.8 and  # High complexity
                len(memory.content) > 500)   # Long content
                
    def _perform_split(self, memory: SolitonMemoryEntry) -> List[SolitonMemoryEntry]:
        """Split memory into components"""
        if len(memory.concept_ids) < 2:
            return []
            
        new_memories = []
        
        # Create new memory for each concept
        for i, concept_id in enumerate(memory.concept_ids):
            new_entry = SolitonMemoryEntry(
                id=f"{memory.id}_split_{i}",
                content=memory.content,  # Could be smarter about content splitting
                memory_type=memory.memory_type,
                phase=self.memory._calculate_concept_phase([concept_id]),
                amplitude=memory.amplitude * 0.8,  # Slightly reduced
                frequency=memory.frequency,
                timestamp=memory.timestamp,
                concept_ids=[concept_id],
                sources=memory.sources,
                metadata={**memory.metadata, "split_from": memory.id},
                polarity=memory.polarity,
                heat=memory.heat * 0.7  # Reduce heat after split
            )
            
            new_memories.append(new_entry)
            
        return new_memories
        
    def _cleanup_empty_oscillators(self):
        """Remove inactive oscillators from lattice"""
        lattice = get_global_lattice()
        
        # Count inactive
        inactive_count = sum(1 for o in lattice.oscillators if not o.get("active", True))
        
        if inactive_count > 0:
            logger.info(f"Cleaned up {inactive_count} inactive oscillators")
