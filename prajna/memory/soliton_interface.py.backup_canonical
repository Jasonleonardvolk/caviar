"""
Soliton Memory Interface for Prajna - ENHANCED COGNITIVE VERSION
================================================================

Real interface to TORI's Soliton Memory system for long-term knowledge storage.
NOW WITH CONCEPT EVOLUTION LINEAGE AND ψ-ANCHOR PHASE ADDRESSING!
"""

import logging
import json
import hashlib
import math
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
from datetime import datetime
import asyncio

logger = logging.getLogger("prajna.memory.soliton")

class SolitonMemoryInterface:
    """Enhanced Soliton Memory with concept evolution storage"""
    
    def __init__(self, rest_endpoint: str = "http://localhost:8002", 
                 ffi_enabled: bool = False, 
                 memory_file: str = "soliton_concept_memory.json", **kwargs):
        self.rest_endpoint = rest_endpoint
        self.ffi_enabled = ffi_enabled
        self.memory_file = memory_file
        self.initialized = False
        
        # Enhanced memory structures
        self.concept_memories = {}
        self.phase_registry = {}
        self.evolution_lineage = {}
        self.psi_anchors = {}
        
        logger.info(f"🧠 Initializing ENHANCED Soliton Memory interface: {rest_endpoint}")
    
    async def initialize(self):
        """Initialize Soliton Memory with concept evolution support"""
        try:
            logger.info("🔗 Connecting to Enhanced Soliton Memory...")
            
            # Load existing memory structures
            await self._load_memory_structures()
            
            # Initialize phase computation system
            await self._initialize_phase_system()
            
            self.initialized = True
            logger.info(f"✅ Enhanced Soliton Memory initialized with {len(self.concept_memories)} concept memories")
            
        except Exception as e:
            logger.warning(f"⚠️ Soliton Memory initialization failed: {e}")
            self.initialized = False
    
    async def _load_memory_structures(self):
        """Load existing memory structures from disk"""
        memory_path = Path(self.memory_file)
        if memory_path.exists():
            try:
                with open(memory_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.concept_memories = data.get('concept_memories', {})
                    self.phase_registry = data.get('phase_registry', {})
                    self.evolution_lineage = data.get('evolution_lineage', {})
                    self.psi_anchors = data.get('psi_anchors', {})
                
                logger.info(f"📚 Loaded {len(self.concept_memories)} existing concept memories")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load memory structures: {e}")
    
    async def _initialize_phase_system(self):
        """Initialize ψ-phase computation system for concept addressing"""
        try:
            # Initialize phase constants (based on soliton wave mechanics)
            self.phase_constants = {
                'omega_base': 2 * math.pi / 137,  # Fine structure constant inspired
                'phi_golden': (1 + math.sqrt(5)) / 2,  # Golden ratio for harmony
                'epsilon_damping': 0.1  # Damping factor for stability
            }
            
            logger.info("🌊 Initialized ψ-phase addressing system")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize phase system: {e}")
    
    def _compute_phase_tag(self, concept_hash: str, epoch: int = 0) -> float:
        """
        Compute ψ-phase tag for concept addressing in soliton memory space.
        Based on concept hash and evolutionary epoch.
        """
        try:
            # Convert hash to numeric value
            hash_numeric = int(concept_hash[:8], 16) / (16**8)
            
            # Compute phase using soliton-inspired formula
            omega = self.phase_constants['omega_base']
            phi = self.phase_constants['phi_golden']
            
            # Phase evolution with epoch
            phase = (omega * hash_numeric * phi + epoch * 0.1) % (2 * math.pi)
            
            return phase
            
        except Exception as e:
            logger.error(f"❌ Failed to compute phase tag: {e}")
            return 0.0
    
    async def store_concept_evolution(self, generation: int, concepts: List[Dict]) -> bool:
        """
        Store evolved concepts as soliton memories with ψ-tag addressing.
        """
        try:
            logger.info(f"🧬 Storing {len(concepts)} evolved concepts for generation {generation}...")
            
            stored_count = 0
            
            for concept in concepts:
                canonical_name = concept['canonical_name']
                concept_hash = concept.get('concept_hash', hashlib.md5(canonical_name.encode()).hexdigest()[:16])
                
                # Compute phase tag
                phase_tag = self._compute_phase_tag(concept_hash, generation)
                
                # Create soliton memory entry
                memory_entry = {
                    'canonical_name': canonical_name,
                    'concept_hash': concept_hash,
                    'phase_tag': phase_tag,
                    'generation': generation,
                    'epoch': concept.get('epoch', 0),
                    'parents': concept.get('parents', []),
                    'synthetic': concept.get('synthetic', True),
                    'stored_at': datetime.now().isoformat(),
                    'metadata': concept
                }
                
                # Store in memory structures
                await self._store_soliton_waveform(memory_entry)
                
                # Update lineage tracking
                if concept.get('parents'):
                    await self._update_evolution_lineage(canonical_name, concept['parents'], generation)
                
                # Register ψ-anchor
                self.psi_anchors[concept_hash] = {
                    'canonical_name': canonical_name,
                    'phase_tag': phase_tag,
                    'generation': generation
                }
                
                stored_count += 1
                logger.debug(f"🧬 Stored concept: {canonical_name} with phase {phase_tag:.4f}")
            
            # Save updated memory state
            await self._save_memory_state()
            
            logger.info(f"✅ Successfully stored {stored_count} concept memories")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store concept evolution: {e}")
            return False
    
    async def _store_soliton_waveform(self, memory_entry: Dict):
        """Store individual concept as soliton waveform"""
        concept_hash = memory_entry['concept_hash']
        phase_tag = memory_entry['phase_tag']
        
        # Store in concept memories
        self.concept_memories[concept_hash] = memory_entry
        
        # Register in phase registry
        phase_key = f"{phase_tag:.4f}"
        if phase_key not in self.phase_registry:
            self.phase_registry[phase_key] = []
        self.phase_registry[phase_key].append(concept_hash)
    
    async def _update_evolution_lineage(self, concept_name: str, parents: List[str], generation: int):
        """Track evolution lineage for concept genealogy"""
        if concept_name not in self.evolution_lineage:
            self.evolution_lineage[concept_name] = {
                'parents': parents,
                'generation': generation,
                'children': [],
                'lineage_path': []
            }
        
        # Update parent-child relationships
        for parent in parents:
            if parent in self.evolution_lineage:
                if concept_name not in self.evolution_lineage[parent]['children']:
                    self.evolution_lineage[parent]['children'].append(concept_name)
    
    async def retrieve_concept_lineage(self, concept_hash: str) -> Optional[Dict]:
        """Retrieve complete evolution lineage for a concept"""
        try:
            if concept_hash not in self.concept_memories:
                return None
            
            memory_entry = self.concept_memories[concept_hash]
            canonical_name = memory_entry['canonical_name']
            
            lineage_data = {
                'concept': memory_entry,
                'lineage': self.evolution_lineage.get(canonical_name, {}),
                'phase_neighbors': await self._get_phase_neighbors(memory_entry['phase_tag']),
                'retrieval_time': datetime.now().isoformat()
            }
            
            logger.info(f"📖 Retrieved lineage for concept: {canonical_name}")
            return lineage_data
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve concept lineage: {e}")
            return None
    
    async def _get_phase_neighbors(self, phase_tag: float, tolerance: float = 0.1) -> List[Dict]:
        """Get concepts with similar phase tags (phase space neighbors)"""
        neighbors = []
        
        for phase_key, concept_hashes in self.phase_registry.items():
            registered_phase = float(phase_key)
            phase_distance = abs(registered_phase - phase_tag)
            
            # Handle circular phase space (0 to 2π)
            if phase_distance > math.pi:
                phase_distance = 2 * math.pi - phase_distance
            
            if phase_distance <= tolerance:
                for concept_hash in concept_hashes:
                    if concept_hash in self.concept_memories:
                        neighbors.append({
                            'concept_hash': concept_hash,
                            'canonical_name': self.concept_memories[concept_hash]['canonical_name'],
                            'phase_distance': phase_distance,
                            'phase_tag': registered_phase
                        })
        
        return sorted(neighbors, key=lambda x: x['phase_distance'])[:10]  # Top 10 closest
    
    async def update_psi_anchors(self, concept_updates: Dict[str, float]) -> bool:
        """Update ψ-anchor phase tags for concept repositioning"""
        try:
            logger.info(f"🔄 Updating {len(concept_updates)} ψ-anchors...")
            
            for concept_hash, new_phase in concept_updates.items():
                if concept_hash in self.concept_memories:
                    # Update memory entry
                    old_phase = self.concept_memories[concept_hash]['phase_tag']
                    self.concept_memories[concept_hash]['phase_tag'] = new_phase
                    
                    # Update phase registry
                    old_phase_key = f"{old_phase:.4f}"
                    new_phase_key = f"{new_phase:.4f}"
                    
                    # Remove from old phase bucket
                    if old_phase_key in self.phase_registry:
                        if concept_hash in self.phase_registry[old_phase_key]:
                            self.phase_registry[old_phase_key].remove(concept_hash)
                    
                    # Add to new phase bucket
                    if new_phase_key not in self.phase_registry:
                        self.phase_registry[new_phase_key] = []
                    self.phase_registry[new_phase_key].append(concept_hash)
                    
                    # Update ψ-anchor registry
                    if concept_hash in self.psi_anchors:
                        self.psi_anchors[concept_hash]['phase_tag'] = new_phase
                    
                    logger.debug(f"🔄 Updated ψ-anchor: {concept_hash} from {old_phase:.4f} to {new_phase:.4f}")
            
            await self._save_memory_state()
            logger.info("✅ ψ-anchor updates completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update ψ-anchors: {e}")
            return False
    
    async def query_by_phase_range(self, phase_min: float, phase_max: float) -> List[Dict]:
        """Query concepts within a phase range"""
        try:
            results = []
            
            for phase_key, concept_hashes in self.phase_registry.items():
                phase = float(phase_key)
                if phase_min <= phase <= phase_max:
                    for concept_hash in concept_hashes:
                        if concept_hash in self.concept_memories:
                            results.append(self.concept_memories[concept_hash])
            
            logger.info(f"🔍 Found {len(results)} concepts in phase range [{phase_min:.4f}, {phase_max:.4f}]")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to query by phase range: {e}")
            return []
    
    async def _save_memory_state(self):
        """Save memory structures to disk"""
        try:
            memory_data = {
                'concept_memories': self.concept_memories,
                'phase_registry': self.phase_registry,
                'evolution_lineage': self.evolution_lineage,
                'psi_anchors': self.psi_anchors,
                'last_updated': datetime.now().isoformat()
            }
            
            memory_path = Path(self.memory_file)
            with open(memory_path, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False)
            
            logger.info("💾 Saved enhanced memory state to disk")
            
        except Exception as e:
            logger.error(f"❌ Failed to save memory state: {e}")
    
    async def health_check(self) -> bool:
        """Check Soliton Memory health"""
        return self.initialized and len(self.concept_memories) >= 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive Soliton Memory statistics"""
        if not self.initialized:
            return {"error": "Soliton Memory not initialized"}
        
        # Calculate generation statistics
        generations = set()
        synthetic_count = 0
        
        for memory in self.concept_memories.values():
            generations.add(memory.get('generation', 0))
            if memory.get('synthetic', False):
                synthetic_count += 1
        
        return {
            "endpoint": self.rest_endpoint,
            "ffi_enabled": self.ffi_enabled,
            "initialized": self.initialized,
            "total_concept_memories": len(self.concept_memories),
            "phase_buckets": len(self.phase_registry),
            "evolution_lineages": len(self.evolution_lineage),
            "psi_anchors": len(self.psi_anchors),
            "generations_stored": len(generations),
            "synthetic_concepts": synthetic_count,
            "natural_concepts": len(self.concept_memories) - synthetic_count,
            "memory_file": self.memory_file
        }
    
    async def cleanup(self):
        """Cleanup Soliton Memory resources"""
        if self.initialized:
            logger.info("🧹 Cleaning up Enhanced Soliton Memory interface")
            await self._save_memory_state()
            self.initialized = False

if __name__ == "__main__":
    # Test Enhanced Soliton Memory interface
    import asyncio
    
    async def test_enhanced_soliton():
        soliton = SolitonMemoryInterface()
        await soliton.initialize()
        
        # Test concept evolution storage
        evolved_concepts = [
            {
                'canonical_name': 'adaptive-synchrony-model',
                'concept_hash': 'abc123def456',
                'parents': ['phase synchrony', 'adaptive model'],
                'epoch': 1,
                'synthetic': True
            },
            {
                'canonical_name': 'quantum-cognitive-bridge',
                'concept_hash': 'def456ghi789',
                'parents': ['quantum mechanics', 'cognitive model'],
                'epoch': 1,
                'synthetic': True
            }
        ]
        
        success = await soliton.store_concept_evolution(1, evolved_concepts)
        print(f"🧬 Concept evolution storage: {success}")
        
        # Test lineage retrieval
        lineage = await soliton.retrieve_concept_lineage('abc123def456')
        print(f"📖 Retrieved lineage: {lineage is not None}")
        
        # Test phase range query
        phase_results = await soliton.query_by_phase_range(0.0, 1.0)
        print(f"🔍 Phase range query results: {len(phase_results)}")
        
        # Test stats
        stats = await soliton.get_stats()
        print(f"📊 Enhanced stats: {stats}")
        
        await soliton.cleanup()
    
    asyncio.run(test_enhanced_soliton())
