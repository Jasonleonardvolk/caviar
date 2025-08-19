#!/usr/bin/env python3
"""
Soliton Memory Integration
Infinite context memory with phase-based retrieval and resonance detection
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import logging
import json
import hashlib
from enum import Enum
from collections import defaultdict

# Import oscillator lattice components
from python.core.oscillator_lattice import get_global_lattice
from python.core.coupling_matrix import CouplingMatrix

from python.core.reasoning_traversal import (
    ConceptNode, ReasoningPath, PrajnaResponsePlus
)
from python.core.unified_metacognitive_integration import (
    SolitonMemorySystem, MetacognitiveContext
)

logger = logging.getLogger(__name__)

# ========== Memory Types and Status ==========

class MemoryType(Enum):
    """Types of memories in the soliton system"""
    SEMANTIC = "semantic"           # Facts and concepts
    EPISODIC = "episodic"          # Experiences and events
    PROCEDURAL = "procedural"       # How-to knowledge
    REFLECTIVE = "reflective"       # Self-generated insights
    TRAUMATIC = "traumatic"         # Problematic/vaulted
    SYNTHETIC = "synthetic"         # AI-generated combinations

class VaultStatus(Enum):
    """Status of memory vaulting"""
    ACTIVE = "active"               # Normal access
    PHASE_45 = "phase_45"          # Light vault (45° shift)
    PHASE_90 = "phase_90"          # Medium vault (90° shift)
    PHASE_180 = "phase_180"        # Deep vault (180° shift)
    QUARANTINE = "quarantine"       # Isolated completely

@dataclass
class SolitonMemoryEntry:
    """Enhanced memory entry with phase and quantum properties"""
    id: str
    content: str
    memory_type: MemoryType
    phase: float
    amplitude: float
    frequency: float
    timestamp: datetime
    concept_ids: List[str]
    sources: List[str]
    vault_status: VaultStatus = VaultStatus.ACTIVE
    resonance_history: List[float] = field(default_factory=list)
    access_count: int = 0
    decay_rate: float = 0.01
    metadata: Dict[str, Any] = field(default_factory=dict)
    polarity: str = "bright"  # "bright" or "dark"
    heat: float = 0.0  # For crystallization
    stability: float = 0.8  # For comfort analysis
    
    def compute_current_amplitude(self) -> float:
        """Compute current amplitude with decay"""
        age_days = (datetime.now(timezone.utc) - self.timestamp).days
        decay_factor = np.exp(-self.decay_rate * age_days)
        
        # Boost for frequently accessed memories
        access_boost = 1.0 + np.log1p(self.access_count) * 0.1
        
        return self.amplitude * decay_factor * access_boost
    
    def compute_phase_drift(self) -> float:
        """Compute phase drift over time"""
        age_hours = (datetime.now(timezone.utc) - self.timestamp).total_seconds() / 3600
        # Natural drift based on frequency
        drift = self.frequency * age_hours * 0.001  # Small drift rate
        return (self.phase + drift) % (2 * np.pi)

# ========== Enhanced Soliton Memory System ==========

class EnhancedSolitonMemory(SolitonMemorySystem):
    """Extended soliton memory with advanced phase mechanics"""
    
    def __init__(self, lattice_size: int = 10000):
        super().__init__(lattice_size)
        self.memory_entries: Dict[str, SolitonMemoryEntry] = {}
        self.phase_lattice = np.zeros((lattice_size,), dtype=complex)
        self.concept_phase_map: Dict[str, float] = {}
        self.resonance_threshold = 0.7
        self.interference_patterns = []
        self.heat_decay_rate = 0.95
        self.heat_boost_on_access = 0.2
        
        # Initialize coupling matrix helper
        self.coupling_helper = CouplingMatrix(100)  # Start with 100 oscillators
    
    def store_enhanced_memory(self, 
                            content: str,
                            concept_ids: List[str],
                            memory_type: MemoryType,
                            sources: List[str],
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store memory with enhanced properties"""
        
        # Generate unique ID
        memory_id = self._generate_memory_id(content, concept_ids)
        
        # Calculate phase based on concepts
        phase = self._calculate_concept_phase(concept_ids)
        
        # Determine frequency from content complexity
        frequency = self._calculate_content_frequency(content)
        
        # Create memory entry
        entry = SolitonMemoryEntry(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            phase=phase,
            amplitude=1.0,
            frequency=frequency,
            timestamp=datetime.now(timezone.utc),
            concept_ids=concept_ids,
            sources=sources,
            metadata=metadata or {}
        )
        
        # Handle dark soliton storage
        if memory_type == MemoryType.TRAUMATIC or metadata.get("suppress", False):
            entry.polarity = "dark"
            self._store_dark_soliton(entry, get_global_lattice())
        else:
            # Store bright soliton normally
            # Store in system
            self.memory_entries[memory_id] = entry
            
            # Update phase lattice
            self._update_phase_lattice(entry)
            
            # Call parent store for compatibility
            super().store_memory(content, concept_ids[0] if concept_ids else "general", 
                               phase, metadata)
            
            # --- NEW: register oscillator in the global lattice --- #
            lattice = get_global_lattice()
            osc_idx = lattice.add_oscillator(
                phase=phase,
                natural_freq=frequency * 0.1,  # Scale frequency down
                amplitude=entry.amplitude,
                stability=1.0,
            )
            
            # Store oscillator index in entry metadata
            entry.metadata["oscillator_idx"] = osc_idx
            
            # Optional: initial weak coupling to recent oscillators
            if len(lattice.oscillators) > 1:
                # Couple to last 5 oscillators with decaying strength
                start_idx = max(0, len(lattice.oscillators) - 6)
                for j in range(start_idx, osc_idx):
                    coupling_strength = 0.02 * (1.0 - (osc_idx - j) / 5.0)
                    lattice.set_coupling(osc_idx, j, coupling_strength)
        
        logger.info(f"Stored enhanced memory {memory_id} with phase {phase:.3f}, polarity {entry.polarity}")
        return memory_id
    
    def _store_dark_soliton(self, entry: SolitonMemoryEntry, lattice):
        """Store dark soliton as baseline + dip oscillators"""
        phase = entry.phase
        freq = entry.frequency * 0.1
        
        # Add baseline oscillator
        base_idx = lattice.add_oscillator(
            phase=phase,
            natural_freq=freq,
            amplitude=1.0,
            stability=1.0
        )
        
        # Add dip oscillator (π out of phase)
        dip_idx = lattice.add_oscillator(
            phase=(phase + np.pi) % (2 * np.pi),
            natural_freq=freq,
            amplitude=entry.amplitude,
            stability=1.0
        )
        
        # Strong coupling between baseline and dip
        lattice.set_coupling(base_idx, dip_idx, 1.0)
        lattice.set_coupling(dip_idx, base_idx, 1.0)
        
        entry.metadata["oscillator_idx"] = dip_idx
        entry.metadata["baseline_idx"] = base_idx
        
        # Store in memory entries
        self.memory_entries[entry.id] = entry
    
    async def create_entity_phase_bond(self, memory_id: str, kb_id: str, bond_strength: float = 1.0):
        """
        Create phase-locked bond between memory and Wikidata entity
        
        Args:
            memory_id: ID of the memory to bond
            kb_id: Wikidata ID (e.g., 'Q42' for Douglas Adams)
            bond_strength: Coupling strength (0-1)
        """
        memory = self.memory_entries.get(memory_id)
        if not memory:
            logger.error(f"Memory {memory_id} not found")
            return False
            
        # Golden ratio phase assignment for entity
        PHI = (1 + np.sqrt(5)) / 2
        try:
            numeric_id = int(kb_id[1:]) if kb_id.startswith('Q') else hash(kb_id)
        except:
            numeric_id = hash(kb_id)
            
        entity_phase = (numeric_id * 2 * np.pi / PHI) % (2 * np.pi)
        
        # Get the global lattice
        lattice = get_global_lattice()
        
        # Check if entity already has an oscillator
        entity_key = f"entity_{kb_id}"
        entity_osc_idx = None
        
        # Search for existing entity oscillator
        if hasattr(self, 'entity_oscillator_map'):
            entity_osc_idx = self.entity_oscillator_map.get(kb_id)
        else:
            self.entity_oscillator_map = {}
        
        # Create entity oscillator if needed
        if entity_osc_idx is None:
            entity_osc_idx = lattice.add_oscillator(
                phase=entity_phase,
                natural_freq=0.1,  # Stable, slow-evolving
                amplitude=0.5,     # Medium strength
                stability=0.9      # High stability for entities
            )
            self.entity_oscillator_map[kb_id] = entity_osc_idx
            logger.info(f"Created entity oscillator for {kb_id} at index {entity_osc_idx}")
        
        # Get memory's oscillator index
        memory_osc_idx = memory.metadata.get("oscillator_idx")
        if memory_osc_idx is None:
            logger.error(f"Memory {memory_id} has no oscillator index")
            return False
        
        # Create bidirectional coupling
        lattice.set_coupling(memory_osc_idx, entity_osc_idx, bond_strength)
        lattice.set_coupling(entity_osc_idx, memory_osc_idx, bond_strength)
        
        # Update memory metadata
        if 'entity_bonds' not in memory.metadata:
            memory.metadata['entity_bonds'] = []
        memory.metadata['entity_bonds'].append({
            'kb_id': kb_id,
            'entity_phase': entity_phase,
            'bond_strength': bond_strength,
            'entity_osc_idx': entity_osc_idx
        })
        
        logger.info(f"Created phase bond: memory {memory_id} <-> entity {kb_id} "
                   f"(phases: {memory.phase:.2f} <-> {entity_phase:.2f} rad)")
        
        return True
    
    def _generate_memory_id(self, content: str, concepts: List[str]) -> str:
        """Generate unique memory ID"""
        combined = content + "".join(sorted(concepts))
        return "sol_" + hashlib.sha256(combined.encode()).hexdigest()[:12]
    
    def _calculate_concept_phase(self, concept_ids: List[str]) -> float:
        """Calculate phase from concept combination"""
        if not concept_ids:
            return np.random.random() * 2 * np.pi
        
        # Use stored phases or generate new ones
        phases = []
        for concept in concept_ids:
            if concept not in self.concept_phase_map:
                # Assign phase based on concept hash
                self.concept_phase_map[concept] = (hash(concept) % 1000) / 1000 * 2 * np.pi
            phases.append(self.concept_phase_map[concept])
        
        # Combine phases through interference
        combined_phase = np.mean(phases)
        return combined_phase % (2 * np.pi)
    
    def _calculate_content_frequency(self, content: str) -> float:
        """Calculate frequency based on content complexity"""
        # Simple heuristic: longer content = lower frequency
        base_freq = 1.0 / (1.0 + len(content) / 1000)
        
        # Adjust for complexity markers
        complexity_markers = ['because', 'therefore', 'however', 'implies']
        complexity_boost = sum(1 for marker in complexity_markers if marker in content.lower())
        
        return base_freq * (1.0 + complexity_boost * 0.1)
    
    def _update_phase_lattice(self, entry: SolitonMemoryEntry):
        """Update the phase lattice with new memory"""
        # Find lattice position from phase
        lattice_pos = int(entry.phase / (2 * np.pi) * self.lattice_size) % self.lattice_size
        
        # Create soliton wave packet
        wave_packet = entry.amplitude * np.exp(1j * entry.phase)
        
        # Add to lattice with interference
        self.phase_lattice[lattice_pos] += wave_packet
        
        # Spread to neighboring positions (quantum tunneling effect)
        spread = 5
        for i in range(1, spread + 1):
            damping = np.exp(-i * 0.5)
            if lattice_pos + i < self.lattice_size:
                self.phase_lattice[lattice_pos + i] += wave_packet * damping * 0.3
            if lattice_pos - i >= 0:
                self.phase_lattice[lattice_pos - i] += wave_packet * damping * 0.3
    
    def find_resonant_memories_enhanced(self, 
                                      query_phase: float,
                                      concept_ids: List[str],
                                      threshold: Optional[float] = None) -> List[SolitonMemoryEntry]:
        """Find memories with enhanced resonance detection"""
        threshold = threshold or self.resonance_threshold
        resonant_memories = []
        
        # First pass: find dark memories that might suppress
        dark_concepts = set()
        for entry_id, entry in self.memory_entries.items():
            if entry.polarity == "dark" and entry.vault_status == VaultStatus.ACTIVE:
                # Check if this dark memory resonates
                resonance = self._calculate_resonance(entry, query_phase, concept_ids)
                if resonance > threshold * 0.5:  # Lower threshold for dark
                    if entry.concept_ids:
                        dark_concepts.add(entry.concept_ids[0])
        
        # Second pass: find bright memories, excluding suppressed ones
        for entry_id, entry in self.memory_entries.items():
            if entry.polarity == "dark":
                continue  # Skip dark memories in results
                
            # Skip vaulted memories unless specifically requested
            if entry.vault_status != VaultStatus.ACTIVE:
                continue
            
            # Skip if suppressed by dark memory
            if entry.concept_ids and entry.concept_ids[0] in dark_concepts:
                continue
            
            # Calculate resonance score
            resonance = self._calculate_resonance(entry, query_phase, concept_ids)
            
            if resonance > threshold:
                # Update access tracking
                entry.access_count += 1
                entry.resonance_history.append(resonance)
                
                # Update heat for crystallization
                entry.heat = min(1.0, entry.heat + self.heat_boost_on_access)
                
                # Keep only recent resonance history
                if len(entry.resonance_history) > 100:
                    entry.resonance_history = entry.resonance_history[-100:]
                
                resonant_memories.append((resonance, entry))
        
        # Sort by resonance strength
        resonant_memories.sort(key=lambda x: x[0], reverse=True)
        
        # Strengthen coupling between resonating memories (Hebbian learning)
        if len(resonant_memories) > 1:
            lattice = get_global_lattice()
            for i in range(min(3, len(resonant_memories) - 1)):
                mem1 = resonant_memories[i][1]
                mem2 = resonant_memories[i + 1][1]
                
                # Get oscillator indices if available
                idx1 = mem1.metadata.get("oscillator_idx")
                idx2 = mem2.metadata.get("oscillator_idx")
                
                if idx1 is not None and idx2 is not None:
                    # Strengthen coupling based on resonance strength
                    strength_delta = 0.01 * resonant_memories[i][0]
                    current = lattice.K[idx1, idx2] if lattice.K is not None else 0
                    if current < 0.5:  # Cap maximum coupling
                        lattice.set_coupling(idx1, idx2, current + strength_delta)
                        lattice.set_coupling(idx2, idx1, current + strength_delta)
        
        # Record interference pattern
        if len(resonant_memories) > 1:
            self._record_interference_pattern([m[1] for m in resonant_memories[:5]])
        
        return [entry for _, entry in resonant_memories]
    
    def _calculate_resonance(self, entry: SolitonMemoryEntry, 
                           query_phase: float,
                           query_concepts: List[str]) -> float:
        """Calculate resonance between memory and query"""
        
        # Phase resonance (with drift compensation)
        current_phase = entry.compute_phase_drift()
        phase_diff = abs(current_phase - query_phase)
        phase_resonance = np.cos(phase_diff) * 0.5 + 0.5  # Normalize to [0, 1]
        
        # Concept overlap resonance
        concept_overlap = len(set(entry.concept_ids) & set(query_concepts))
        concept_resonance = concept_overlap / max(len(entry.concept_ids), len(query_concepts)) if query_concepts else 0
        
        # Amplitude factor (stronger memories resonate more)
        amplitude_factor = entry.compute_current_amplitude()
        
        # Frequency matching (similar complexity)
        freq_match = 1.0 / (1.0 + abs(entry.frequency - 0.5))  # Assume query freq ~0.5
        
        # Combined resonance
        resonance = (phase_resonance * 0.4 + 
                    concept_resonance * 0.3 + 
                    amplitude_factor * 0.2 + 
                    freq_match * 0.1)
        
        return resonance
    
    def _record_interference_pattern(self, memories: List[SolitonMemoryEntry]):
        """Record interference patterns between resonating memories"""
        pattern = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "memory_ids": [m.id for m in memories],
            "phases": [m.compute_phase_drift() for m in memories],
            "interference_type": self._classify_interference(memories)
        }
        
        self.interference_patterns.append(pattern)
        
        # Keep only recent patterns
        if len(self.interference_patterns) > 1000:
            self.interference_patterns = self.interference_patterns[-1000:]
    
    def _classify_interference(self, memories: List[SolitonMemoryEntry]) -> str:
        """Classify the type of interference between memories"""
        if len(memories) < 2:
            return "none"
        
        # Check phase relationships
        phases = [m.compute_phase_drift() for m in memories]
        phase_diffs = [abs(phases[i] - phases[i+1]) for i in range(len(phases)-1)]
        avg_diff = np.mean(phase_diffs)
        
        if avg_diff < np.pi / 4:
            return "constructive"  # Similar phases
        elif avg_diff > 3 * np.pi / 4:
            return "destructive"  # Opposite phases
        else:
            return "mixed"  # Various phases
    
    def detect_memory_dissonance_enhanced(self, 
                                        new_memory: SolitonMemoryEntry,
                                        related_memories: List[SolitonMemoryEntry]) -> Dict[str, Any]:
        """Enhanced dissonance detection with phase analysis"""
        
        dissonance_factors = {
            "content_contradiction": 0.0,
            "phase_opposition": 0.0,
            "temporal_inconsistency": 0.0,
            "source_conflict": 0.0
        }
        
        for related in related_memories:
            # Content contradiction (semantic)
            if self._contradicts_content(new_memory.content, related.content):
                dissonance_factors["content_contradiction"] += 0.3
            
            # Phase opposition
            phase_diff = abs(new_memory.phase - related.compute_phase_drift())
            if phase_diff > np.pi * 0.8:  # Nearly opposite phases
                dissonance_factors["phase_opposition"] += 0.2
            
            # Temporal inconsistency
            if new_memory.memory_type == MemoryType.EPISODIC and related.memory_type == MemoryType.EPISODIC:
                if self._temporal_conflict(new_memory, related):
                    dissonance_factors["temporal_inconsistency"] += 0.3
            
            # Source conflict
            if self._source_conflict(new_memory.sources, related.sources):
                dissonance_factors["source_conflict"] += 0.2
        
        # Normalize factors
        num_comparisons = len(related_memories)
        for key in dissonance_factors:
            dissonance_factors[key] /= max(1, num_comparisons)
        
        # Calculate overall dissonance
        total_dissonance = sum(dissonance_factors.values()) / len(dissonance_factors)
        
        return {
            "total_dissonance": total_dissonance,
            "factors": dissonance_factors,
            "recommendation": self._get_dissonance_recommendation(total_dissonance)
        }
    
    def _contradicts_content(self, content1: str, content2: str) -> bool:
        """Check if contents contradict each other"""
        # Simple heuristic - could be enhanced with NLP
        negation_words = ['not', 'never', 'false', 'incorrect', 'wrong']
        
        c1_lower = content1.lower()
        c2_lower = content2.lower()
        
        # Check for explicit negation
        for neg in negation_words:
            if (neg in c1_lower and neg not in c2_lower) or \
               (neg in c2_lower and neg not in c1_lower):
                return True
        
        return False
    
    def _temporal_conflict(self, mem1: SolitonMemoryEntry, mem2: SolitonMemoryEntry) -> bool:
        """Check for temporal conflicts in episodic memories"""
        # If memories claim to be at same time but different events
        time1 = mem1.metadata.get("event_time")
        time2 = mem2.metadata.get("event_time")
        
        if time1 and time2 and abs((time1 - time2).total_seconds()) < 300:  # Within 5 minutes
            if mem1.content != mem2.content:
                return True
        
        return False
    
    def _source_conflict(self, sources1: List[str], sources2: List[str]) -> bool:
        """Check if sources are known to conflict"""
        # Simple check - could maintain a conflict matrix
        if "wikipedia" in str(sources1).lower() and "arxiv" in str(sources2).lower():
            return True  # Different reliability levels
        
        return False
    
    def _get_dissonance_recommendation(self, dissonance: float) -> str:
        """Get recommendation based on dissonance level"""
        if dissonance < 0.2:
            return "Harmonious - proceed with confidence"
        elif dissonance < 0.5:
            return "Minor dissonance - consider reconciliation"
        elif dissonance < 0.7:
            return "Significant dissonance - requires resolution"
        else:
            return "High dissonance - consider vaulting conflicting memories"
    
    def vault_memory_with_phase_shift(self, memory_id: str, 
                                    vault_status: VaultStatus,
                                    reason: str):
        """Vault memory with appropriate phase shift"""
        if memory_id not in self.memory_entries:
            logger.warning(f"Memory {memory_id} not found for vaulting")
            return
        
        entry = self.memory_entries[memory_id]
        
        # Determine phase shift based on vault status
        phase_shifts = {
            VaultStatus.PHASE_45: np.pi / 4,
            VaultStatus.PHASE_90: np.pi / 2,
            VaultStatus.PHASE_180: np.pi,
            VaultStatus.QUARANTINE: np.random.random() * 2 * np.pi  # Random phase
        }
        
        shift = phase_shifts.get(vault_status, 0)
        
        # Apply phase shift
        entry.phase = (entry.phase + shift) % (2 * np.pi)
        entry.vault_status = vault_status
        entry.metadata["vault_reason"] = reason
        entry.metadata["vault_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Update phase lattice
        self._update_phase_lattice(entry)
        
        logger.info(f"Vaulted memory {memory_id} with {vault_status.value}")
    
    def nightly_crystallization(self) -> Dict[str, Any]:
        """Perform nightly memory crystallization with fusion purge"""
        logger.info("=== Starting Nightly Memory Crystallization with Fusion Purge ===")
        
        report = {
            "migrated": 0,
            "decayed": 0,
            "fused": 0,
            "split": 0,
            "purged": 0
        }
        
        # 1. Update heat values (decay)
        for entry in self.memory_entries.values():
            entry.heat *= self.heat_decay_rate
            
        # 2. Sort memories by heat
        sorted_memories = sorted(
            self.memory_entries.items(),
            key=lambda x: x[1].heat,
            reverse=True
        )
        
        # 3. Migrate hot memories to stable positions
        lattice = get_global_lattice()
        hot_threshold = 0.7
        cold_threshold = 0.1
        
        for i, (mem_id, entry) in enumerate(sorted_memories):
            if entry.heat > hot_threshold and i > len(sorted_memories) // 2:
                # Hot memory in unstable position - migrate
                self._migrate_to_stable_position(entry, lattice)
                report["migrated"] += 1
                
            elif entry.heat < cold_threshold:
                # Cold memory - allow to decay
                entry.amplitude *= 0.9
                if entry.amplitude < 0.1:
                    # Remove very weak memories
                    self._remove_memory(mem_id, lattice)
                    report["decayed"] += 1
        
        # 4. Perform fusion and fission
        report["fused"] = self._perform_memory_fusion()
        report["split"] = self._perform_memory_fission()
        
        # 5. FUSION PURGE: Run maintenance pass to clean up flagged oscillators
        if hasattr(lattice, 'maintenance_pass'):
            maintenance_report = lattice.maintenance_pass()
            report["purged"] = maintenance_report["removed"]
            logger.info(f"Fusion purge: Removed {maintenance_report['removed']} inactive oscillators")
        
        logger.info(f"Crystallization complete with fusion purge: {report}")
        return report
    
    def _migrate_to_stable_position(self, entry: SolitonMemoryEntry, lattice):
        """Migrate memory to more stable lattice position"""
        if "oscillator_idx" in entry.metadata:
            old_idx = entry.metadata["oscillator_idx"]
            
            # In Kagome topology, center positions are more stable
            # This is simplified - real implementation would use topology info
            new_idx = old_idx  # Placeholder
            
            # Update stability
            entry.stability = min(1.0, entry.stability + 0.1)
            
    def _perform_memory_fusion(self) -> int:
        """Fuse similar memories"""
        fused = 0
        concept_groups = defaultdict(list)
        
        # Group by primary concept
        for mem_id, entry in self.memory_entries.items():
            if entry.concept_ids:
                concept_groups[entry.concept_ids[0]].append((mem_id, entry))
        
        # Fuse duplicates
        for concept, memories in concept_groups.items():
            if len(memories) > 1:
                # Sort by amplitude
                memories.sort(key=lambda x: x[1].amplitude, reverse=True)
                main_id, main_entry = memories[0]
                
                # Merge others into main
                for other_id, other_entry in memories[1:]:
                    if self._should_fuse(main_entry, other_entry):
                        # Combine amplitudes
                        main_entry.amplitude = min(2.0, 
                            np.sqrt(main_entry.amplitude**2 + other_entry.amplitude**2))
                        
                        # Merge heat
                        main_entry.heat = max(main_entry.heat, other_entry.heat)
                        
                        # Remove other
                        self.memory_entries.pop(other_id, None)
                        fused += 1
        
        return fused
    
    def _should_fuse(self, mem1: SolitonMemoryEntry, mem2: SolitonMemoryEntry) -> bool:
        """Determine if two memories should fuse"""
        # Same concept and similar content
        if mem1.concept_ids == mem2.concept_ids:
            # Could add content similarity check here
            return True
        return False
    
    def _perform_memory_fission(self) -> int:
        """Split complex memories"""
        # Placeholder - would split overly complex memories
        return 0
    
    def _remove_memory(self, memory_id: str, lattice):
        """Remove memory from system with proper oscillator cleanup"""
        if memory_id in self.memory_entries:
            entry = self.memory_entries[memory_id]
            
            # FUSION PURGE: Properly remove oscillator and shrink matrix
            if "oscillator_idx" in entry.metadata:
                idx = entry.metadata["oscillator_idx"]
                
                # Call the proper removal that recycles memory & shrinks Laplacian
                if hasattr(lattice, 'remove_oscillator'):
                    lattice.remove_oscillator(idx)
                    logger.info(f"Fusion purge: Removed oscillator {idx}, matrix shrunk")
                else:
                    # Fallback: mark inactive (temporary until maintenance pass)
                    if idx < len(lattice.oscillators):
                        lattice.oscillators[idx].amplitude = 0.0  # Zero amplitude neutralizes
                        logger.warning(f"Oscillator {idx} flagged inactive - needs maintenance pass")
                        
                        # Schedule for removal in next maintenance pass
                        if not hasattr(lattice, '_pending_removals'):
                            lattice._pending_removals = set()
                        lattice._pending_removals.add(idx)
            
            # Handle baseline oscillator for dark solitons
            if "baseline_idx" in entry.metadata:
                baseline_idx = entry.metadata["baseline_idx"]
                if hasattr(lattice, 'remove_oscillator'):
                    lattice.remove_oscillator(baseline_idx)
                    logger.info(f"Fusion purge: Removed baseline oscillator {baseline_idx}")
            
            # Remove from entries
            del self.memory_entries[memory_id]
            logger.info(f"Fusion purge: Memory {memory_id} completely removed")
    
    def consolidate_memories(self, time_window_days: int = 7) -> Dict[str, Any]:
        """Consolidate similar memories within time window"""
        
        consolidation_report = {
            "consolidated_groups": [],
            "total_consolidated": 0,
            "space_saved": 0
        }
        
        # Group memories by concept overlap within time window
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=time_window_days)
        recent_memories = [
            entry for entry in self.memory_entries.values()
            if entry.timestamp > cutoff_time and entry.vault_status == VaultStatus.ACTIVE
        ]
        
        # Find groups to consolidate
        groups = self._find_consolidation_groups(recent_memories)
        
        for group in groups:
            if len(group) > 1:
                # Create consolidated memory
                consolidated = self._create_consolidated_memory(group)
                
                # Store consolidated memory
                consolidated_id = self.store_enhanced_memory(
                    consolidated["content"],
                    consolidated["concepts"],
                    MemoryType.SYNTHETIC,
                    consolidated["sources"],
                    {"consolidation_from": [m.id for m in group]}
                )
                
                # Vault original memories
                for memory in group:
                    self.vault_memory_with_phase_shift(
                        memory.id, 
                        VaultStatus.PHASE_45,
                        f"Consolidated into {consolidated_id}"
                    )
                
                consolidation_report["consolidated_groups"].append({
                    "new_id": consolidated_id,
                    "original_count": len(group),
                    "concepts": consolidated["concepts"]
                })
        
        consolidation_report["total_consolidated"] += len(group) - 1
        
        return consolidation_report
    
    def _find_consolidation_groups(self, memories: List[SolitonMemoryEntry]) -> List[List[SolitonMemoryEntry]]:
        """Find groups of memories that can be consolidated"""
        groups = []
        used = set()
        
        for i, mem1 in enumerate(memories):
            if mem1.id in used:
                continue
            
            group = [mem1]
            used.add(mem1.id)
            
            for j, mem2 in enumerate(memories[i+1:], i+1):
                if mem2.id in used:
                    continue
                
                # Check similarity
                concept_overlap = len(set(mem1.concept_ids) & set(mem2.concept_ids))
                if concept_overlap >= len(mem1.concept_ids) * 0.7:  # 70% overlap
                    group.append(mem2)
                    used.add(mem2.id)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _create_consolidated_memory(self, group: List[SolitonMemoryEntry]) -> Dict[str, Any]:
        """Create a consolidated memory from a group"""
        # Combine contents
        contents = [m.content for m in group]
        consolidated_content = "Consolidated insight: " + " Additionally, ".join(contents)
        
        # Union of concepts
        all_concepts = set()
        for m in group:
            all_concepts.update(m.concept_ids)
        
        # Union of sources
        all_sources = set()
        for m in group:
            all_sources.update(m.sources)
        
        return {
            "content": consolidated_content,
            "concepts": list(all_concepts),
            "sources": list(all_sources)
        }

# ========== Soliton Memory Integration ==========

class SolitonMemoryIntegration:
    """Integrates soliton memory with reasoning system"""
    
    def __init__(self, memory_system: EnhancedSolitonMemory):
        self.memory = memory_system
        self.resonance_cache = {}
        self.dissonance_threshold = 0.6
    
    def store_reasoning_path(self, path: ReasoningPath, 
                           query: str,
                           response: PrajnaResponsePlus) -> str:
        """Store a reasoning path as a soliton memory"""
        
        # Extract concepts from path
        concept_ids = [node.id for node in path.chain]
        
        # Create content from path
        content = self._create_path_content(path, query, response.text)
        
        # Gather sources
        sources = []
        for node in path.chain:
            sources.extend(node.sources)
        sources = list(set(sources))  # Unique sources
        
        # Determine memory type
        memory_type = self._classify_memory_type(path)
        
        # Store in soliton memory
        memory_id = self.memory.store_enhanced_memory(
            content=content,
            concept_ids=concept_ids,
            memory_type=memory_type,
            sources=sources,
            metadata={
                "query": query,
                "path_type": path.path_type,
                "confidence": path.confidence,
                "response_preview": response.text[:200]
            }
        )
        
        return memory_id
    
    def _create_path_content(self, path: ReasoningPath, query: str, response: str) -> str:
        """Create memory content from reasoning path"""
        chain_str = " → ".join([n.name for n in path.chain])
        
        content = f"Query: {query}\n"
        content += f"Reasoning: {chain_str}\n"
        content += f"Justifications: {', '.join(path.edge_justifications)}\n"
        content += f"Response: {response[:200]}..."
        
        return content
    
    def _classify_memory_type(self, path: ReasoningPath) -> MemoryType:
        """Classify memory type based on reasoning path"""
        if path.path_type == "causal":
            return MemoryType.SEMANTIC
        elif path.path_type == "support":
            return MemoryType.PROCEDURAL
        elif "reflect" in str(path.metadata):
            return MemoryType.REFLECTIVE
        else:
            return MemoryType.SEMANTIC
    
    def check_memory_consistency(self, new_response: PrajnaResponsePlus) -> Dict[str, Any]:
        """Check if new response is consistent with memory"""
        
        # Extract concepts from response
        concepts = []
        for path in new_response.reasoning_paths:
            concepts.extend([n.id for n in path.chain])
        concepts = list(set(concepts))
        
        # Calculate query phase
        query_phase = self.memory._calculate_concept_phase(concepts)
        
        # Find resonant memories
        resonant = self.memory.find_resonant_memories_enhanced(
            query_phase, concepts, threshold=0.5
        )
        
        if not resonant:
            return {
                "consistent": True,
                "dissonance": 0.0,
                "message": "No related memories found"
            }
        
        # Create temporary memory entry for comparison
        temp_memory = SolitonMemoryEntry(
            id="temp",
            content=new_response.text,
            memory_type=MemoryType.SEMANTIC,
            phase=query_phase,
            amplitude=1.0,
            frequency=0.5,
            timestamp=datetime.now(timezone.utc),
            concept_ids=concepts,
            sources=new_response.sources
        )
        
        # Check dissonance
        dissonance_report = self.memory.detect_memory_dissonance_enhanced(
            temp_memory, resonant[:5]
        )
        
        # Determine consistency
        consistent = dissonance_report["total_dissonance"] < self.dissonance_threshold
        
        return {
            "consistent": consistent,
            "dissonance": dissonance_report["total_dissonance"],
            "factors": dissonance_report["factors"],
            "related_memories": len(resonant),
            "recommendation": dissonance_report["recommendation"],
            "conflicting_memories": self._identify_conflicts(temp_memory, resonant) if not consistent else []
        }
    
    def _identify_conflicts(self, new_memory: SolitonMemoryEntry, 
                          related: List[SolitonMemoryEntry]) -> List[Dict[str, Any]]:
        """Identify specific conflicts with existing memories"""
        conflicts = []
        
        for memory in related[:3]:  # Top 3 conflicts
            if self.memory._contradicts_content(new_memory.content, memory.content):
                conflicts.append({
                    "memory_id": memory.id,
                    "type": "content_contradiction",
                    "existing": memory.content[:100],
                    "new": new_memory.content[:100]
                })
        
        return conflicts
    
    def retrieve_supporting_memories(self, 
                                   reasoning_paths: List[ReasoningPath],
                                   limit: int = 5) -> List[SolitonMemoryEntry]:
        """Retrieve memories that support the reasoning"""
        
        # Collect all concepts
        all_concepts = []
        for path in reasoning_paths:
            all_concepts.extend([n.id for n in path.chain])
        
        # Calculate phase
        query_phase = self.memory._calculate_concept_phase(all_concepts)
        
        # Find resonant memories
        supporting = self.memory.find_resonant_memories_enhanced(
            query_phase, all_concepts, threshold=0.6
        )
        
        # Filter for truly supporting memories
        filtered = []
        for memory in supporting:
            # Check if memory supports rather than contradicts
            if not self._likely_contradicts(memory, reasoning_paths):
                filtered.append(memory)
        
        return filtered[:limit]
    
    def _likely_contradicts(self, memory: SolitonMemoryEntry, 
                          paths: List[ReasoningPath]) -> bool:
        """Check if memory likely contradicts the reasoning"""
        # Simple heuristic - check for negation words
        negation_indicators = ['not', 'false', 'incorrect', 'wrong', 'never']
        
        memory_has_negation = any(neg in memory.content.lower() for neg in negation_indicators)
        
        # Check if paths mention similar concepts but with negation
        for path in paths:
            for node in path.chain:
                if any(concept in memory.concept_ids for concept in [node.id]):
                    path_has_negation = any(neg in node.description.lower() for neg in negation_indicators)
                    
                    if memory_has_negation != path_has_negation:
                        return True
        
        return False
    
    def get_memory_analytics(self) -> Dict[str, Any]:
        """Get analytics on memory usage and patterns"""
        
        total_memories = len(self.memory.memory_entries)
        
        # Type distribution
        type_dist = defaultdict(int)
        for entry in self.memory.memory_entries.values():
            type_dist[entry.memory_type.value] += 1
        
        # Vault distribution
        vault_dist = defaultdict(int)
        for entry in self.memory.memory_entries.values():
            vault_dist[entry.vault_status.value] += 1
        
        # Phase distribution (quantized)
        phase_dist = defaultdict(int)
        for entry in self.memory.memory_entries.values():
            phase_bucket = int(entry.phase / (np.pi / 4))  # 8 buckets
            phase_dist[f"phase_{phase_bucket}"] += 1
        
        # Recent access patterns
        recently_accessed = sorted(
            [e for e in self.memory.memory_entries.values() if e.access_count > 0],
            key=lambda e: e.access_count,
            reverse=True
        )[:10]
        
        # Interference patterns
        interference_types = defaultdict(int)
        for pattern in self.memory.interference_patterns[-100:]:
            interference_types[pattern["interference_type"]] += 1
        
        return {
            "total_memories": total_memories,
            "type_distribution": dict(type_dist),
            "vault_distribution": dict(vault_dist),
            "phase_distribution": dict(phase_dist),
            "top_accessed": [
                {
                    "id": m.id,
                    "access_count": m.access_count,
                    "concepts": m.concept_ids[:3]
                }
                for m in recently_accessed
            ],
            "interference_patterns": dict(interference_types),
            "lattice_utilization": np.count_nonzero(self.memory.phase_lattice) / self.memory.lattice_size,
            "oscillator_coherence": get_global_lattice().order_parameter(),
            "oscillator_entropy": get_global_lattice().phase_entropy(),
            "oscillator_count": len(get_global_lattice().oscillators)
        }

# ========== Demo and Testing ==========

def demonstrate_soliton_memory():
    """Demonstrate soliton memory integration"""
    
    print("🌊 Soliton Memory Integration Demo")
    print("=" * 60)
    
    # Initialize memory system
    memory_system = EnhancedSolitonMemory()
    integration = SolitonMemoryIntegration(memory_system)
    
    # Create test reasoning paths
    test_paths = [
        ReasoningPath(
            chain=[
                ConceptNode("entropy", "Entropy", "Measure of uncertainty"),
                ConceptNode("information", "Information", "Reduces uncertainty"),
                ConceptNode("compression", "Compression", "Encodes information")
            ],
            edge_justifications=["implies", "enables"],
            score=0.9,
            path_type="causal",
            confidence=0.85
        )
    ]
    
    test_response = PrajnaResponsePlus(
        text="Entropy measures uncertainty, which information reduces, enabling compression.",
        reasoning_paths=test_paths,
        sources=["info_theory_2024"],
        confidence=0.85
    )
    
    # Test 1: Store reasoning path
    print("\n1️⃣ Storing reasoning path as memory...")
    memory_id = integration.store_reasoning_path(
        test_paths[0],
        "How does entropy relate to compression?",
        test_response
    )
    print(f"   Stored with ID: {memory_id}")
    
    # Test 2: Check consistency with similar response
    print("\n2️⃣ Checking consistency with similar response...")
    similar_response = PrajnaResponsePlus(
        text="Entropy quantifies uncertainty in information theory.",
        reasoning_paths=test_paths,
        sources=["shannon_1948"],
        confidence=0.9
    )
    
    consistency = integration.check_memory_consistency(similar_response)
    print(f"   Consistent: {consistency['consistent']}")
    print(f"   Dissonance: {consistency['dissonance']:.3f}")
    print(f"   Recommendation: {consistency['recommendation']}")
    
    # Test 3: Check with contradictory response
    print("\n3️⃣ Checking consistency with contradictory response...")
    
    contradictory_paths = [
        ReasoningPath(
            chain=[
                ConceptNode("entropy", "Entropy", "Not related to information"),
                ConceptNode("randomness", "Randomness", "Pure chaos")
            ],
            edge_justifications=["implies"],
            score=0.6,
            path_type="inference",
            confidence=0.5
        )
    ]
    
    contradictory_response = PrajnaResponsePlus(
        text="Entropy is not related to information theory.",
        reasoning_paths=contradictory_paths,
        sources=["dubious_blog"],
        confidence=0.5
    )
    
    consistency2 = integration.check_memory_consistency(contradictory_response)
    print(f"   Consistent: {consistency2['consistent']}")
    print(f"   Dissonance: {consistency2['dissonance']:.3f}")
    print(f"   Factors: {consistency2['factors']}")
    
    # Test 4: Retrieve supporting memories
    print("\n4️⃣ Retrieving supporting memories...")
    supporting = integration.retrieve_supporting_memories(test_paths, limit=3)
    print(f"   Found {len(supporting)} supporting memories")
    for mem in supporting:
        print(f"   - {mem.id}: {mem.content[:50]}...")
    
    # Test 5: Memory analytics
    print("\n5️⃣ Memory Analytics")
    analytics = integration.get_memory_analytics()
    print(f"   Total memories: {analytics['total_memories']}")
    print(f"   Type distribution: {analytics['type_distribution']}")
    print(f"   Phase distribution: {analytics['phase_distribution']}")
    print(f"   Lattice utilization: {analytics['lattice_utilization']:.2%}")
    
    # Test 6: Consolidation
    print("\n6️⃣ Testing memory consolidation...")
    
    # Add more memories for consolidation
    for i in range(3):
        memory_system.store_enhanced_memory(
            f"Entropy variant {i}: measures information uncertainty",
            ["entropy", "information"],
            MemoryType.SEMANTIC,
            [f"source_{i}"]
        )
    
    consolidation = memory_system.consolidate_memories(time_window_days=1)
    print(f"   Consolidated {consolidation['total_consolidated']} memories")
    print(f"   Groups: {len(consolidation['consolidated_groups'])}")
    
    # Test 7: Dark soliton suppression
    print("\n7️⃣ Testing dark soliton suppression...")
    
    # Store bright memory
    bright_id = memory_system.store_enhanced_memory(
        "The sky is blue",
        ["sky", "color"],
        MemoryType.SEMANTIC,
        ["observation"]
    )
    
    # Store dark memory to suppress
    dark_id = memory_system.store_enhanced_memory(
        "Forget about sky color",
        ["sky"],
        MemoryType.TRAUMATIC,
        ["suppression"]
    )
    
    # Try to recall - should be suppressed
    query_phase = memory_system._calculate_concept_phase(["sky"])
    results = memory_system.find_resonant_memories_enhanced(query_phase, ["sky"])
    print(f"   Memories found after suppression: {len(results)}")
    
    # Test 8: Nightly crystallization with fusion purge
    print("\n8️⃣ Testing nightly crystallization with fusion purge...")
    
    # Check oscillator count before
    lattice = get_global_lattice()
    before_count = len(lattice.oscillators)
    print(f"   Oscillators before crystallization: {before_count}")
    
    report = memory_system.nightly_crystallization()
    
    # Check oscillator count after
    after_count = len(lattice.oscillators)
    print(f"   Oscillators after crystallization: {after_count}")
    print(f"   Crystallization report: {report}")
    print(f"   Matrix size change: {before_count} -> {after_count} (saved {before_count - after_count} slots)")
    
    # Test 9: Manual fusion purge
    print("\n9️⃣ Testing manual oscillator removal...")
    
    if len(lattice.oscillators) > 0:
        # Remove an oscillator manually
        removed_idx = len(lattice.oscillators) - 1
        success = lattice.remove_oscillator(removed_idx)
        print(f"   Manual removal of oscillator {removed_idx}: {'SUCCESS' if success else 'FAILED'}")
        print(f"   Oscillators after manual removal: {len(lattice.oscillators)}")
        
        # Test maintenance pass
        lattice._pending_removals = {0} if len(lattice.oscillators) > 0 else set()
        maintenance = lattice.maintenance_pass()
        print(f"   Maintenance pass result: {maintenance}")
    
    print("\n✅ Soliton memory integration with fusion purge complete!")
    print("\n🔧 FUSION PURGE FEATURES:")
    print("   - Inactive oscillators properly removed from matrix")
    print("   - Matrix rows/cols shrunk to recycle memory")
    print("   - Maintenance pass during crystallization")
    print("   - Index stability maintained during removal")
    print("   - No more ghost oscillators occupying space!")

if __name__ == "__main__":
    demonstrate_soliton_memory()
