"""
Test script for the modernized spaCy entity extraction pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.quality import extract_and_boost_concepts, analyze_concept_purity, reset_frequency_counter

def test_spacy_extraction():
    """Test the new spaCy-based concept extraction"""
    
    print("🧪 Testing spaCy Entity Extraction Pipeline")
    print("=" * 60)
    
    # Test text with known entities
    test_chunks = [
        {
            "text": "Albert Einstein developed the theory of relativity in 1905. His work at Princeton University revolutionized physics.",
            "section": "introduction"
        },
        {
            "text": "Douglas Adams wrote The Hitchhiker's Guide to the Galaxy. The answer to life, universe and everything is 42.",
            "section": "body"
        },
        {
            "text": "The Large Hadron Collider at CERN discovered the Higgs boson. This confirmed predictions from quantum field theory.",
            "section": "results"
        }
    ]
    
    all_concepts = []
    
    # Process each chunk
    for i, chunk_data in enumerate(test_chunks):
        print(f"\n📄 Processing chunk {i+1} ({chunk_data['section']}):")
        print(f"   Text: {chunk_data['text'][:60]}...")
        
        concepts = extract_and_boost_concepts(
            chunk=chunk_data['text'],
            chunk_index=i,
            chunk_section=chunk_data['section'],
            title_text="Famous Scientists and Their Discoveries",
            abstract_text="A review of groundbreaking scientific achievements"
        )
        
        print(f"   Found {len(concepts)} concepts")
        
        # Show entity-linked concepts
        entity_linked = [c for c in concepts if 'wikidata_id' in c.get('metadata', {})]
        if entity_linked:
            print(f"   🔗 Entity-linked concepts:")
            for concept in entity_linked:
                print(f"      - {concept['name']} → {concept['metadata']['wikidata_id']} "
                      f"(confidence: {concept['score']:.2f})")
        
        all_concepts.extend(concepts)
    
    # Analyze purity
    print("\n🔬 Analyzing concept purity...")
    pure_concepts = analyze_concept_purity(
        all_concepts,
        doc_name="test_document.pdf",
        title_text="Famous Scientists and Their Discoveries",
        abstract_text="A review of groundbreaking scientific achievements"
    )
    
    print(f"\n✅ Results:")
    print(f"   Total concepts extracted: {len(all_concepts)}")
    print(f"   Pure concepts after filtering: {len(pure_concepts)}")
    
    # Show top concepts
    print(f"\n🏆 Top 10 concepts by quality score:")
    for i, concept in enumerate(pure_concepts[:10]):
        kb_info = ""
        if 'wikidata_id' in concept.get('metadata', {}):
            kb_info = f" [{concept['metadata']['wikidata_id']}]"
        print(f"   {i+1}. {concept['name']}{kb_info} "
              f"(quality: {concept.get('quality_score', 0):.3f}, method: {concept['method']})")
    
    # Show extraction methods used
    methods = set(c['method'] for c in all_concepts)
    print(f"\n📊 Extraction methods used: {', '.join(methods)}")
    
    # Reset frequency counter for next document
    reset_frequency_counter()
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_spacy_extraction()
