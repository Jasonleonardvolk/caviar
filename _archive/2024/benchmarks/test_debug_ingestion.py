# test_debug_ingestion.py - Debug what's happening
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def debug_ingestion():
    """Debug the ingestion process step by step"""
    
    # Create test file
    test_file = "debug_test.txt"
    test_content = "This is a simple test document for debugging the ingestion pipeline."
    
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    try:
        print("🔍 Step 1: Testing file extraction...")
        from core.universal_file_extractor import extract_content_from_file
        content = await extract_content_from_file(test_file)
        print(f"✅ Extracted content: {content[:50]}...")
        
        print("\n🔍 Step 2: Testing concept extraction...")
        from core.concept_extractor_enhanced import ProductionConceptExtractor
        extractor = ProductionConceptExtractor()
        concepts = await extractor.extract_concepts(content)
        print(f"✅ Extracted {len(concepts)} concepts")
        for i, concept in enumerate(concepts[:3]):
            print(f"  - Concept {i+1}: {concept.text[:50]}...")
        
        print("\n🔍 Step 3: Testing embedding generation...")
        from core.embedding_client_noauth import embed_concepts
        concept_texts = [c.text for c in concepts]
        embeddings = await embed_concepts(concept_texts)
        print(f"✅ Generated embeddings: shape {embeddings.shape}")
        
        print("\n🔍 Step 4: Testing Penrose verification...")
        from core.penrose_verifier_test import get_penrose_verifier
        verifier = get_penrose_verifier()
        result = verifier.verify_tessera(embeddings, concept_texts)
        print(f"✅ Penrose result: {result.status}")
        
        print("\n✅ All components working! Issue might be in the integration.")
        
    except Exception as e:
        print(f"\n❌ Error at current step: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    asyncio.run(debug_ingestion())
