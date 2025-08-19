"""
ConceptMesh Entity Linking Integration Example

Shows how to use the updated ConceptMesh with Wikidata entity linking
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.core.concept_mesh import ConceptMesh

def demo_entity_linking():
    """Demonstrate entity linking features in ConceptMesh"""
    
    print("🧠 ConceptMesh Entity Linking Demo")
    print("=" * 60)
    
    # Initialize mesh
    mesh = ConceptMesh.instance()
    
    # Example 1: Add concepts with Wikidata IDs
    print("\n1️⃣ Adding entity-linked concepts...")
    
    # Add Einstein with KB link
    einstein_id = mesh.add_concept_from_kb(
        name="Albert Einstein",
        kb_id="Q937",
        entity_type="PERSON",
        confidence=0.95,
        description="Theoretical physicist who developed the theory of relativity",
        entity_phase=2.834,  # From phase calculation
        phase_locked=True
    )
    print(f"   Added Einstein: {einstein_id}")
    
    # Add Douglas Adams with KB link
    adams_id = mesh.add_concept_from_kb(
        name="Douglas Adams",
        kb_id="Q42",
        entity_type="PERSON",
        confidence=0.98,
        description="British author of The Hitchhiker's Guide to the Galaxy",
        entity_phase=1.293,
        phase_locked=True
    )
    print(f"   Added Adams: {adams_id}")
    
    # Example 2: Try to add duplicate KB entity
    print("\n2️⃣ Testing KB deduplication...")
    duplicate_id = mesh.add_concept_from_kb(
        name="A. Einstein",  # Different name
        kb_id="Q937",  # Same KB ID
        entity_type="PERSON",
        confidence=0.90
    )
    print(f"   Duplicate attempt returned: {duplicate_id}")
    print(f"   Same as original? {duplicate_id == einstein_id}")
    
    # Example 3: Find by KB ID
    print("\n3️⃣ Finding concepts by KB ID...")
    found = mesh.find_concept_by_kb_id("Q42")
    if found:
        print(f"   Found by KB ID Q42: {found.name}")
        print(f"   Phase locked: {found.metadata.get('phase_locked', False)}")
        print(f"   Entity phase: {found.metadata.get('entity_phase', 'N/A')}")
    
    # Example 4: Add regular concept then link to KB
    print("\n4️⃣ Linking existing concept to KB...")
    relativity_id = mesh.add_concept(
        name="Theory of Relativity",
        description="Einstein's groundbreaking physics theory",
        category="physics"
    )
    
    # Later, link it to Wikidata
    success = mesh.link_concept_to_kb(
        concept_id=relativity_id,
        kb_id="Q11455",
        entity_phase=3.721,
        phase_locked=True
    )
    print(f"   Linked relativity to KB: {success}")
    
    # Example 5: Add concepts with relationships
    print("\n5️⃣ Adding related concepts...")
    
    physics_id = mesh.add_concept_from_kb(
        name="Physics",
        kb_id="Q413",
        entity_type="FIELD",
        confidence=0.99,
        entity_phase=0.892,
        phase_locked=True
    )
    
    # Create relationships
    mesh.add_relation(einstein_id, physics_id, "contributed_to", strength=0.95)
    mesh.add_relation(relativity_id, physics_id, "part_of", strength=0.90)
    mesh.add_relation(einstein_id, relativity_id, "created", strength=1.0)
    
    # Example 6: Get all canonical concepts
    print("\n6️⃣ Listing all canonical concepts...")
    canonical = mesh.get_canonical_concepts()
    print(f"   Total canonical concepts: {len(canonical)}")
    for concept in canonical[:5]:  # Show first 5
        print(f"   - {concept.name} [{concept.metadata.get('wikidata_id', 'N/A')}]")
    
    # Example 7: Statistics
    print("\n7️⃣ Mesh statistics...")
    stats = mesh.get_statistics()
    print(f"   Total concepts: {stats['total_concepts']}")
    print(f"   Canonical concepts: {stats.get('canonical_concepts', 0)}")
    print(f"   Total relations: {stats['total_relations']}")
    
    # Example 8: Phase coherence check
    print("\n8️⃣ Checking phase coherence...")
    related = mesh.get_related_concepts(einstein_id)
    for rel_id, rel_type, strength in related:
        if rel_id in mesh.concepts:
            rel_concept = mesh.concepts[rel_id]
            if 'entity_phase' in rel_concept.metadata:
                print(f"   {rel_concept.name}: phase={rel_concept.metadata['entity_phase']:.3f}")
    
    print("\n✅ Entity linking demo complete!")

def demo_ingestion_integration():
    """Show how ingestion pipeline integrates with ConceptMesh"""
    
    print("\n\n🔄 Ingestion Integration Example")
    print("=" * 60)
    
    mesh = ConceptMesh.instance()
    
    # Simulate concepts coming from spaCy entity linker
    extracted_entities = [
        {
            "name": "CERN",
            "wikidata_id": "Q42944",
            "entity_type": "ORG",
            "confidence": 0.92,
            "entity_phase": 4.567,
            "section": "introduction"
        },
        {
            "name": "Large Hadron Collider",
            "wikidata_id": "Q1333",
            "entity_type": "FACILITY",
            "confidence": 0.88,
            "entity_phase": 2.103,
            "section": "methodology"
        },
        {
            "name": "Higgs boson",
            "wikidata_id": "Q402",
            "entity_type": "PARTICLE",
            "confidence": 0.95,
            "entity_phase": 0.761,
            "section": "results"
        }
    ]
    
    print("\n📥 Processing extracted entities...")
    concept_ids = []
    
    for entity in extracted_entities:
        concept_id = mesh.add_concept_from_kb(
            name=entity["name"],
            kb_id=entity["wikidata_id"],
            entity_type=entity["entity_type"],
            confidence=entity["confidence"],
            entity_phase=entity["entity_phase"],
            phase_locked=True,
            additional_metadata={"section": entity["section"]}
        )
        concept_ids.append(concept_id)
        print(f"   Stored: {entity['name']} -> {entity['wikidata_id']}")
    
    # Add relationships based on co-occurrence
    print("\n🔗 Creating relationships...")
    mesh.add_relation(concept_ids[0], concept_ids[1], "operates", strength=0.9)
    mesh.add_relation(concept_ids[1], concept_ids[2], "discovered", strength=0.95)
    
    print("\n✅ Ingestion integration complete!")

if __name__ == "__main__":
    demo_entity_linking()
    demo_ingestion_integration()
