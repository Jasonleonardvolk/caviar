"""
Entity Phase Bond Testing Script
Tests the create_entity_phase_bond functionality
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.core.soliton_memory_integration import EnhancedSolitonMemory, MemoryType

async def test_entity_phase_bond():
    """Test entity phase bonding functionality"""
    
    print("🔗 Testing Entity Phase Bond Creation")
    print("=" * 60)
    
    # Initialize memory system
    memory_system = EnhancedSolitonMemory()
    
    # Test case 1: Store a memory about Douglas Adams
    print("\n1️⃣ Storing memory about Douglas Adams...")
    memory_id = memory_system.store_enhanced_memory(
        content="Douglas Adams was a British author best known for The Hitchhiker's Guide to the Galaxy",
        concept_ids=["douglas_adams", "author", "science_fiction"],
        memory_type=MemoryType.SEMANTIC,
        sources=["wikipedia"],
        metadata={"wikidata_id": "Q42"}  # Douglas Adams' Wikidata ID
    )
    print(f"   Memory stored: {memory_id}")
    
    # Test case 2: Create entity phase bond
    print("\n2️⃣ Creating entity phase bond...")
    success = await memory_system.create_entity_phase_bond(
        memory_id=memory_id,
        kb_id="Q42",
        bond_strength=1.0
    )
    print(f"   Phase bond created: {success}")
    
    # Test case 3: Verify bond in memory metadata
    print("\n3️⃣ Verifying phase bond in metadata...")
    memory = memory_system.memory_entries[memory_id]
    if 'entity_bonds' in memory.metadata:
        bonds = memory.metadata['entity_bonds']
        print(f"   Entity bonds found: {len(bonds)}")
        for bond in bonds:
            print(f"   - KB ID: {bond['kb_id']}")
            print(f"   - Entity Phase: {bond['entity_phase']:.4f} rad")
            print(f"   - Bond Strength: {bond['bond_strength']}")
            print(f"   - Entity Oscillator Index: {bond['entity_osc_idx']}")
    
    # Test case 4: Store another memory about same entity
    print("\n4️⃣ Storing another memory about same entity...")
    memory_id2 = memory_system.store_enhanced_memory(
        content="The answer to life, universe and everything is 42 - from Douglas Adams",
        concept_ids=["douglas_adams", "42", "humor"],
        memory_type=MemoryType.SEMANTIC,
        sources=["hitchhikers_guide"],
        metadata={"wikidata_id": "Q42"}
    )
    
    success2 = await memory_system.create_entity_phase_bond(
        memory_id=memory_id2,
        kb_id="Q42",
        bond_strength=0.8
    )
    print(f"   Second memory stored and bonded: {success2}")
    
    # Test case 5: Check that same entity oscillator is reused
    print("\n5️⃣ Verifying entity oscillator reuse...")
    mem1_bonds = memory_system.memory_entries[memory_id].metadata.get('entity_bonds', [])
    mem2_bonds = memory_system.memory_entries[memory_id2].metadata.get('entity_bonds', [])
    
    if mem1_bonds and mem2_bonds:
        osc_idx1 = mem1_bonds[0]['entity_osc_idx']
        osc_idx2 = mem2_bonds[0]['entity_osc_idx']
        print(f"   Memory 1 entity oscillator: {osc_idx1}")
        print(f"   Memory 2 entity oscillator: {osc_idx2}")
        print(f"   Same oscillator reused: {osc_idx1 == osc_idx2}")
    
    # Test case 6: Different entity
    print("\n6️⃣ Testing with different entity (Einstein)...")
    memory_id3 = memory_system.store_enhanced_memory(
        content="Albert Einstein developed the theory of relativity",
        concept_ids=["einstein", "physics", "relativity"],
        memory_type=MemoryType.SEMANTIC,
        sources=["physics_textbook"],
        metadata={"wikidata_id": "Q937"}  # Einstein's Wikidata ID
    )
    
    success3 = await memory_system.create_entity_phase_bond(
        memory_id=memory_id3,
        kb_id="Q937",
        bond_strength=1.0
    )
    
    mem3_bonds = memory_system.memory_entries[memory_id3].metadata.get('entity_bonds', [])
    if mem3_bonds:
        print(f"   Einstein's entity phase: {mem3_bonds[0]['entity_phase']:.4f} rad")
        print(f"   Einstein's oscillator index: {mem3_bonds[0]['entity_osc_idx']}")
    
    # Test case 7: Check global lattice state
    print("\n7️⃣ Checking global lattice state...")
    from python.core.oscillator_lattice import get_global_lattice
    lattice = get_global_lattice()
    print(f"   Total oscillators: {len(lattice.oscillators)}")
    print(f"   Active oscillators: {sum(1 for o in lattice.oscillators if o.get('active', True))}")
    
    # Show coupling matrix for entity oscillators
    if hasattr(memory_system, 'entity_oscillator_map'):
        print(f"   Entity oscillators tracked: {len(memory_system.entity_oscillator_map)}")
        print(f"   Entities: {list(memory_system.entity_oscillator_map.keys())}")
    
    print("\n✅ Entity phase bond testing complete!")

if __name__ == "__main__":
    asyncio.run(test_entity_phase_bond())
