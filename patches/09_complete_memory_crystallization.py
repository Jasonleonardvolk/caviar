#!/usr/bin/env python3
"""
Complete implementation of memory_crystallization.py
Handles the crystallization process during nightly maintenance
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class CrystallizationConfig:
    """Configuration for memory crystallization"""
    hot_threshold: float = 0.7
    cold_threshold: float = 0.1
    migration_stability_boost: float = 0.1
    fusion_threshold: float = 0.7
    fusion_similarity_threshold: float = 0.7
    stable_zone_start: float = 0.33  # Start of stable zone (33%)
    stable_zone_end: float = 0.67    # End of stable zone (67%)
    heat_decay_rate: float = 0.95
    max_migrations_per_run: int = 100
    max_fusions_per_run: int = 50

class MemoryCrystallizer:
    """
    Handles memory crystallization - the process of organizing memories
    based on their heat (access patterns) and relationships
    """
    
    def __init__(self, config: Optional[CrystallizationConfig] = None):
        self.config = config or CrystallizationConfig()
        self.last_run_time = None
        self.run_history = []
        
    def crystallize(self, memory_system, lattice) -> Dict[str, Any]:
        """
        Main crystallization process
        Returns report of actions taken
        """
        start_time = datetime.now(timezone.utc)
        logger.info("=== Starting Memory Crystallization ===")
        
        report = {
            "start_time": start_time,
            "initial_state": self._capture_state(memory_system),
            "migrations": 0,
            "fusions": 0,
            "decayed": 0,
            "errors": []
        }
        
        try:
            # Step 1: Update heat values (decay)
            self._decay_heat(memory_system)
            
            # Step 2: Analyze memory distribution
            distribution = self._analyze_distribution(memory_system, lattice)
            
            # Step 3: Migrate hot memories to stable positions
            migration_report = self._migrate_memories(memory_system, lattice, distribution)
            report["migrations"] = migration_report["migrated"]
            report["migration_details"] = migration_report
            
            # Step 4: Fuse similar memories
            fusion_report = self._fuse_memories(memory_system, lattice)
            report["fusions"] = fusion_report["fused"]
            report["fusion_details"] = fusion_report
            
            # Step 5: Decay cold memories
            decay_report = self._decay_cold_memories(memory_system, lattice)
            report["decayed"] = decay_report["decayed"]
            report["decay_details"] = decay_report
            
            # Step 6: Optimize coupling matrix
            self._optimize_couplings(memory_system, lattice)
            
        except Exception as e:
            logger.error(f"Crystallization error: {e}")
            report["errors"].append(str(e))
        
        # Finalize report
        end_time = datetime.now(timezone.utc)
        report["end_time"] = end_time
        report["duration"] = (end_time - start_time).total_seconds()
        report["final_state"] = self._capture_state(memory_system)
        
        # Store in history
        self.last_run_time = start_time
        self.run_history.append(report)
        if len(self.run_history) > 30:
            self.run_history = self.run_history[-30:]
        
        logger.info(f"=== Crystallization Complete: "
                   f"{report['migrations']} migrations, "
                   f"{report['fusions']} fusions, "
                   f"{report['decayed']} decayed ===")
        
        return report
    
    def _capture_state(self, memory_system) -> Dict[str, Any]:
        """Capture current state metrics"""
        total_memories = len(memory_system.memory_entries)
        total_heat = sum(m.heat for m in memory_system.memory_entries.values())
        avg_heat = total_heat / max(1, total_memories)
        
        heat_distribution = defaultdict(int)
        for memory in memory_system.memory_entries.values():
            bucket = int(memory.heat * 10)  # 0-10 buckets
            heat_distribution[bucket] += 1
        
        return {
            "total_memories": total_memories,
            "average_heat": avg_heat,
            "heat_distribution": dict(heat_distribution)
        }
    
    def _decay_heat(self, memory_system):
        """Apply heat decay to all memories"""
        for memory in memory_system.memory_entries.values():
            memory.heat *= self.config.heat_decay_rate
            
            # Ensure minimum heat for recently accessed
            if memory.access_count > 0:
                days_old = (datetime.now(timezone.utc) - memory.timestamp).days
                if days_old < 7:  # Recent memories maintain some heat
                    memory.heat = max(memory.heat, 0.1)
    
    def _analyze_distribution(self, memory_system, lattice) -> Dict[str, Any]:
        """Analyze current memory distribution"""
        memories_by_position = defaultdict(list)
        
        # Group memories by their oscillator positions
        for mem_id, memory in memory_system.memory_entries.items():
            if "oscillator_idx" in memory.metadata:
                idx = memory.metadata["oscillator_idx"]
                memories_by_position[idx].append((mem_id, memory))
        
        # Calculate position stability scores
        total_positions = len(lattice.oscillators) if hasattr(lattice, 'oscillators') else 1000
        stable_start = int(total_positions * self.config.stable_zone_start)
        stable_end = int(total_positions * self.config.stable_zone_end)
        
        position_scores = {}
        for idx in range(total_positions):
            if stable_start <= idx <= stable_end:
                position_scores[idx] = 1.0  # Stable zone
            else:
                # Less stable as we move away from center
                distance_from_stable = min(
                    abs(idx - stable_start),
                    abs(idx - stable_end)
                )
                position_scores[idx] = 1.0 / (1.0 + distance_from_stable * 0.01)
        
        return {
            "memories_by_position": dict(memories_by_position),
            "position_scores": position_scores,
            "stable_zone": (stable_start, stable_end),
            "total_positions": total_positions
        }
    
    def _migrate_memories(self, memory_system, lattice, distribution) -> Dict[str, Any]:
        """Migrate hot memories to stable positions"""
        report = {
            "migrated": 0,
            "migrations": []
        }
        
        stable_start, stable_end = distribution["stable_zone"]
        position_scores = distribution["position_scores"]
        
        # Find hot memories in unstable positions
        migration_candidates = []
        
        for mem_id, memory in memory_system.memory_entries.items():
            if memory.heat > self.config.hot_threshold:
                if "oscillator_idx" in memory.metadata:
                    current_idx = memory.metadata["oscillator_idx"]
                    current_score = position_scores.get(current_idx, 0)
                    
                    if current_score < 0.8:  # Not in optimal position
                        migration_candidates.append({
                            "memory_id": mem_id,
                            "memory": memory,
                            "current_idx": current_idx,
                            "current_score": current_score,
                            "heat": memory.heat
                        })
        
        # Sort by heat (hottest first)
        migration_candidates.sort(key=lambda x: x["heat"], reverse=True)
        
        # Perform migrations
        for candidate in migration_candidates[:self.config.max_migrations_per_run]:
            new_idx = self._find_stable_position(
                lattice, stable_start, stable_end, candidate["current_idx"]
            )
            
            if new_idx is not None and new_idx != candidate["current_idx"]:
                # Perform migration
                success = self._perform_migration(
                    memory_system, lattice, 
                    candidate["memory"], 
                    candidate["current_idx"], 
                    new_idx
                )
                
                if success:
                    report["migrated"] += 1
                    report["migrations"].append({
                        "memory_id": candidate["memory_id"],
                        "from": candidate["current_idx"],
                        "to": new_idx,
                        "heat": candidate["heat"]
                    })
                    
                    # Boost stability
                    candidate["memory"].stability = min(
                        1.0, 
                        candidate["memory"].stability + self.config.migration_stability_boost
                    )
        
        return report
    
    def _find_stable_position(self, lattice, stable_start, stable_end, current_idx) -> Optional[int]:
        """Find an available stable position"""
        # Try to find an inactive oscillator in stable zone
        if hasattr(lattice, 'oscillators'):
            # First, try positions near the center
            center = (stable_start + stable_end) // 2
            search_radius = (stable_end - stable_start) // 2
            
            for distance in range(0, search_radius, 10):
                for direction in [1, -1]:
                    idx = center + (distance * direction)
                    if stable_start <= idx <= stable_end:
                        if idx < len(lattice.oscillators):
                            osc = lattice.oscillators[idx]
                            if not osc.get("active", True) or osc.get("amplitude", 1.0) < 0.1:
                                return idx
        
        return None
    
    def _perform_migration(self, memory_system, lattice, memory, old_idx, new_idx) -> bool:
        """Perform the actual migration"""
        try:
            if hasattr(lattice, 'oscillators'):
                # Copy oscillator data
                if old_idx < len(lattice.oscillators) and new_idx < len(lattice.oscillators):
                    lattice.oscillators[new_idx] = lattice.oscillators[old_idx].copy()
                    lattice.oscillators[old_idx]["active"] = False
                    lattice.oscillators[old_idx]["amplitude"] = 0.0
                    
                    # Update memory metadata
                    memory.metadata["oscillator_idx"] = new_idx
                    memory.metadata["migration_time"] = datetime.now(timezone.utc).isoformat()
                    
                    logger.debug(f"Migrated memory from position {old_idx} to {new_idx}")
                    return True
        except Exception as e:
            logger.error(f"Migration failed: {e}")
        
        return False
    
    def _fuse_memories(self, memory_system, lattice) -> Dict[str, Any]:
        """Fuse similar memories to save space"""
        report = {
            "fused": 0,
            "fusion_groups": []
        }
        
        # Group memories by primary concept
        concept_groups = defaultdict(list)
        for mem_id, memory in memory_system.memory_entries.items():
            if memory.concept_ids:
                primary_concept = memory.concept_ids[0]
                concept_groups[primary_concept].append((mem_id, memory))
        
        # Find fusion candidates
        for concept, memories in concept_groups.items():
            if len(memories) < 2:
                continue
            
            # Sort by amplitude (strongest first)
            memories.sort(key=lambda x: x[1].amplitude, reverse=True)
            
            # Try to fuse similar memories
            fusion_count = 0
            i = 0
            while i < len(memories) - 1 and fusion_count < self.config.max_fusions_per_run:
                main_id, main_memory = memories[i]
                
                j = i + 1
                while j < len(memories):
                    other_id, other_memory = memories[j]
                    
                    if self._should_fuse(main_memory, other_memory):
                        # Perform fusion
                        self._perform_fusion(
                            memory_system, lattice,
                            main_memory, other_memory,
                            main_id, other_id
                        )
                        
                        # Remove fused memory from list
                        memories.pop(j)
                        fusion_count += 1
                        report["fused"] += 1
                        
                        report["fusion_groups"].append({
                            "concept": concept,
                            "main": main_id,
                            "absorbed": other_id,
                            "new_amplitude": main_memory.amplitude
                        })
                    else:
                        j += 1
                
                i += 1
        
        return report
    
    def _should_fuse(self, mem1, mem2) -> bool:
        """Determine if two memories should be fused"""
        # Don't fuse if they were recently split
        if "split_from" in mem1.metadata or "split_from" in mem2.metadata:
            return False
        
        # Check content similarity
        if hasattr(mem1, 'content') and hasattr(mem2, 'content'):
            similarity = self._calculate_similarity(mem1.content, mem2.content)
            if similarity < self.config.fusion_similarity_threshold:
                return False
        
        # Check concept match
        if set(mem1.concept_ids) != set(mem2.concept_ids):
            return False
        
        # Check heat levels - don't fuse hot memories
        if mem1.heat > 0.5 and mem2.heat > 0.5:
            return False
        
        return True
    
    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity (simple word overlap)"""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _perform_fusion(self, memory_system, lattice, main_memory, other_memory,
                       main_id, other_id):
        """Perform memory fusion"""
        # Combine properties
        main_memory.amplitude = min(2.0, np.sqrt(
            main_memory.amplitude**2 + other_memory.amplitude**2
        ))
        main_memory.heat = max(main_memory.heat, other_memory.heat)
        main_memory.access_count += other_memory.access_count
        
        # Mark oscillator as inactive
        if "oscillator_idx" in other_memory.metadata:
            idx = other_memory.metadata["oscillator_idx"]
            if hasattr(lattice, 'oscillators') and idx < len(lattice.oscillators):
                lattice.oscillators[idx]["active"] = False
        
        # Remove the fused memory
        memory_system.memory_entries.pop(other_id, None)
        
        logger.debug(f"Fused memory {other_id} into {main_id}")
    
    def _decay_cold_memories(self, memory_system, lattice) -> Dict[str, Any]:
        """Decay and potentially remove cold memories"""
        report = {
            "decayed": 0,
            "removed": []
        }
        
        memories_to_remove = []
        
        for mem_id, memory in memory_system.memory_entries.items():
            if memory.heat < self.config.cold_threshold:
                # Apply decay to amplitude
                memory.amplitude *= 0.9
                
                # Remove if amplitude too low
                if memory.amplitude < 0.1:
                    memories_to_remove.append(mem_id)
        
        # Remove dead memories
        for mem_id in memories_to_remove:
            if mem_id in memory_system.memory_entries:
                memory = memory_system.memory_entries[mem_id]
                
                # Mark oscillator as inactive
                if "oscillator_idx" in memory.metadata:
                    idx = memory.metadata["oscillator_idx"]
                    if hasattr(lattice, 'oscillators') and idx < len(lattice.oscillators):
                        lattice.oscillators[idx]["active"] = False
                
                # Remove memory
                del memory_system.memory_entries[mem_id]
                report["decayed"] += 1
                report["removed"].append(mem_id)
        
        return report
    
    def _optimize_couplings(self, memory_system, lattice):
        """Optimize coupling matrix based on crystallization results"""
        if not hasattr(lattice, 'coupling_matrix'):
            return
        
        # Remove very weak couplings
        weak_threshold = 0.001
        keys_to_remove = []
        
        for key, value in lattice.coupling_matrix.items():
            if abs(value) < weak_threshold:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del lattice.coupling_matrix[key]
        
        logger.debug(f"Removed {len(keys_to_remove)} weak couplings")
    
    def get_status(self) -> Dict[str, Any]:
        """Get crystallizer status"""
        recent_runs = []
        for run in self.run_history[-5:]:
            recent_runs.append({
                "timestamp": run["start_time"].isoformat(),
                "duration": run["duration"],
                "migrations": run["migrations"],
                "fusions": run["fusions"],
                "decayed": run["decayed"]
            })
        
        return {
            "last_run": self.last_run_time.isoformat() if self.last_run_time else None,
            "total_runs": len(self.run_history),
            "recent_runs": recent_runs,
            "config": {
                "hot_threshold": self.config.hot_threshold,
                "cold_threshold": self.config.cold_threshold,
                "fusion_threshold": self.config.fusion_threshold
            }
        }
