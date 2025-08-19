# test_direct_pipeline.py - Test each component directly
import asyncio

async def test_components():
    print("🔍 Testing pipeline components directly...\n")
    
    # Test text
    test_text = "This is a test document about quantum computing and machine learning."
    
    # Test 1: Concept extraction
    print("1️⃣ Testing concept extraction...")
    try:
        from core.concept_extractor_full import FullTextConceptExtractor
        extractor = FullTextConceptExtractor(max_concepts_per_chunk=5)
        concepts = await extractor.extract_concepts(test_text)
        print(f"✅ Extracted {len(concepts)} concepts")
        for i, c in enumerate(concepts):
            print(f"   - Concept {i+1}: {c.text}")
    except Exception as e:
        print(f"❌ Concept extraction failed: {e}")
        return
    
    # Test 2: Embedding
    print("\n2️⃣ Testing embedding client...")
    try:
        from core.embedding_client_noauth import SimpleEmbeddingClient
        client = SimpleEmbeddingClient()
        
        # Test with simple text
        test_texts = ["quantum computing", "machine learning"]
        result = await client.embed_texts(test_texts)
        
        print(f"✅ Got embeddings: shape {result.vectors.shape}")
        await client.close()
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Direct HTTP call
    print("\n3️⃣ Testing direct HTTP embedding...")
    try:
        import httpx
        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(
                "http://localhost:8080/embed",
                json={"texts": ["test concept"]},
                timeout=30.0
            )
            print(f"✅ HTTP response: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   - Got {len(data['embeddings'])} embeddings")
    except Exception as e:
        print(f"❌ HTTP call failed: {e}")
    
    print("\n✅ Component testing complete!")

if __name__ == "__main__":
    asyncio.run(test_components())
