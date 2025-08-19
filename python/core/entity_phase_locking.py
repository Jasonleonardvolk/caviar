"""
Entity-Linked Phase Locking for Soliton Memory
Connects Wikidata entity links to phase-locked memory bonds
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EntityBond:
    """Phase-locked bond between entities"""
    entity1_id: str  # Wikidata ID
    entity2_id: str  # Wikidata ID
    phase_offset: float  # Relative phase in radians
    bond_strength: float  # Coupling strength
    semantic_distance: float  # From knowledge graph

class EntityPhaseLocking:
    """
    Creates unbreakable semantic bonds via phase locking
    Uses Wikidata IDs from entity linker to establish phase relationships
    """
    
    def __init__(self, memory_system, concept_mesh):
        self.memory = memory_system
        self.mesh = concept_mesh
        
        # Golden ratio for quasi-periodic phase separation
        self.PHI = (1 + np.sqrt(5)) / 2
        
        # Entity phase map: Wikidata ID -> phase
        self.entity_phases = {}
        
        # Phase bonds between entities
        self.phase_bonds = []
        
    def kb_id_to_phase(self, kb_id: str) -> float:
        """
        Convert Wikidata ID to unique phase using golden ratio
        Ensures quasi-periodic non-repeating phases
        """
        if kb_id in self.entity_phases:
            return self.entity_phases[kb_id]
            
        # Extract numeric part of Wikidata ID (e.g., Q42 -> 42)
        try:
            numeric_id = int(kb_id[1:]) if kb_id.startswith('Q') else hash(kb_id)
        except:
            numeric_id = hash(kb_id)
            
        # Golden ratio phase assignment
        # This creates maximally separated phases
        phase = (numeric_id * 2 * np.pi / self.PHI) % (2 * np.pi)
        
        self.entity_phases[kb_id] = phase
        return phase
    
    async def create_phase_bond(self, 
                               memory_id: str,
                               entity_kb_id: str,
                               bond_strength: float = 1.0):
        """
        Create phase-locked bond between memory and entity
        """
        # Get memory's current phase
        memory_entry = self.memory.memory_entries.get(memory_id)
        if not memory_entry:
            logger.error(f"Memory {memory_id} not found")
            return
            
        memory_phase = memory_entry.phase
        entity_phase = self.kb_id_to_phase(entity_kb_id)
        
        # Calculate phase offset for locking
        phase_offset = (entity_phase - memory_phase) % (2 * np.pi)
        
        # Update oscillator coupling in lattice
        if hasattr(memory_entry, 'oscillator_index'):
            osc_idx = memory_entry.oscillator_index
            
            # Find or create entity oscillator
            entity_osc_idx = await self._get_or_create_entity_oscillator(entity_kb_id)
            
            # Set coupling strength (bidirectional)
            lattice = self.memory.lattice
            lattice.set_coupling(osc_idx, entity_osc_idx, bond_strength)
            lattice.set_coupling(entity_osc_idx, osc_idx, bond_strength)
            
            logger.info(f"Created phase bond: memory {memory_id} <-> entity {entity_kb_id} "
                       f"(offset: {phase_offset:.2f} rad)")
    
    async def link_related_entities(self, entities: List[Dict[str, str]]):
        """
        Create phase bonds between related entities from same context
        Uses semantic distance from knowledge graph
        """
        # Group entities that appear together
        for i, entity1 in enumerate(entities):
            kb_id1 = entity1.get('wikidata_id')
            if not kb_id1:
                continue
                
            for entity2 in entities[i+1:]:
                kb_id2 = entity2.get('wikidata_id')
                if not kb_id2:
                    continue
                    
                # Calculate semantic distance (simplified)
                # In practice, would query knowledge graph
                distance = self._estimate_semantic_distance(kb_id1, kb_id2)
                
                if distance < 0.5:  # Closely related
                    bond = EntityBond(
                        entity1_id=kb_id1,
                        entity2_id=kb_id2,
                        phase_offset=self._calculate_optimal_offset(kb_id1, kb_id2),
                        bond_strength=1.0 - distance,
                        semantic_distance=distance
                    )
                    self.phase_bonds.append(bond)
                    
                    # Create oscillator coupling
                    await self._couple_entity_oscillators(bond)
    
    def _calculate_optimal_offset(self, kb_id1: str, kb_id2: str) -> float:
        """
        Calculate optimal phase offset to prevent interference
        Uses number theory to ensure non-repeating patterns
        """
        phase1 = self.kb_id_to_phase(kb_id1)
        phase2 = self.kb_id_to_phase(kb_id2)
        
        # Natural offset
        natural_offset = (phase2 - phase1) % (2 * np.pi)
        
        # Adjust if too close (risk of interference)
        if natural_offset < np.pi/6 or natural_offset > 11*np.pi/6:
            # Use golden angle for maximum separation
            natural_offset = 2 * np.pi / self.PHI
            
        return natural_offset
    
    def _estimate_semantic_distance(self, kb_id1: str, kb_id2: str) -> float:
        """
        Estimate semantic distance between entities
        In production, would query Wikidata SPARQL endpoint
        """
        # Simplified heuristic based on ID numbers
        try:
            num1 = int(kb_id1[1:]) if kb_id1.startswith('Q') else 0
            num2 = int(kb_id2[1:]) if kb_id2.startswith('Q') else 0
            
            # Closer ID numbers often indicate related concepts
            # This is a vast oversimplification!
            id_distance = abs(num1 - num2) / 1000000
            return min(1.0, id_distance)
        except:
            return 0.5  # Default medium distance
    
    async def _get_or_create_entity_oscillator(self, kb_id: str) -> int:
        """Get or create oscillator for entity"""
        # Check if entity already has oscillator
        entity_key = f"entity_{kb_id}"
        
        # Search in memory entries
        for memory_id, entry in self.memory.memory_entries.items():
            if entry.metadata.get('entity_kb_id') == kb_id:
                return entry.oscillator_index
                
        # Create new oscillator
        phase = self.kb_id_to_phase(kb_id)
        lattice = self.memory.lattice
        
        osc_idx = lattice.add_oscillator(
            phase=phase,
            natural_freq=0.1,  # Slow evolution for entities
            amplitude=0.5,     # Medium strength
            stability=0.9      # High stability
        )
        
        # Store mapping
        self.memory.entity_oscillator_map[kb_id] = osc_idx
        
        return osc_idx
    
    async def _couple_entity_oscillators(self, bond: EntityBond):
        """Create coupling between entity oscillators"""
        osc1 = await self._get_or_create_entity_oscillator(bond.entity1_id)
        osc2 = await self._get_or_create_entity_oscillator(bond.entity2_id)
        
        lattice = self.memory.lattice
        lattice.set_coupling(osc1, osc2, bond.bond_strength)
        lattice.set_coupling(osc2, osc1, bond.bond_strength)
        
        logger.info(f"Coupled entities {bond.entity1_id} <-> {bond.entity2_id} "
                   f"(strength: {bond.bond_strength:.2f})")
    
    def analyze_phase_network(self) -> Dict[str, Any]:
        """Analyze the phase-locked entity network"""
        if not self.phase_bonds:
            return {'status': 'No phase bonds created'}
            
        # Analyze phase distribution
        phases = list(self.entity_phases.values())
        phase_separations = []
        
        for i in range(len(phases)):
            for j in range(i+1, len(phases)):
                sep = abs(phases[i] - phases[j])
                sep = min(sep, 2*np.pi - sep)  # Wrap around
                phase_separations.append(sep)
        
        return {
            'total_entities': len(self.entity_phases),
            'total_bonds': len(self.phase_bonds),
            'avg_phase_separation': np.mean(phase_separations) if phase_separations else 0,
            'min_phase_separation': np.min(phase_separations) if phase_separations else 0,
            'interference_risk': 'low' if np.min(phase_separations) > np.pi/6 else 'high',
            'quasi_periodic': True  # Always true with golden ratio assignment
        }


# Integration with memory sculptor
async def sculpt_with_entity_linking(sculptor, user_id: str, raw_concept: Dict[str, Any]):
    """
    Enhanced sculpting that creates phase bonds for linked entities
    """
    # Extract entities from metadata
    entities = []
    concept_metadata = raw_concept.get('metadata', {})
    
    # Check if this concept has Wikidata ID
    if 'wikidata_id' in concept_metadata:
        entities.append({
            'name': raw_concept.get('name', ''),
            'wikidata_id': concept_metadata['wikidata_id'],
            'wikidata_url': concept_metadata.get('wikidata_url', '')
        })
    
    # Store memory as usual
    memory_ids = await sculptor.sculpt_and_store(user_id, raw_concept)
    
    # Create phase bonds if entities found
    if entities and memory_ids:
        phase_locker = EntityPhaseLocking(sculptor.memory, sculptor.mesh)
        
        for memory_id in memory_ids:
            for entity in entities:
                await phase_locker.create_phase_bond(
                    memory_id,
                    entity['wikidata_id']
                )
        
        # Link related entities
        await phase_locker.link_related_entities(entities)
        
        # Analyze network
        analysis = phase_locker.analyze_phase_network()
        logger.info(f"Phase network: {analysis}")
    
    return memory_ids
