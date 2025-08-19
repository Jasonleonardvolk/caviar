# debug_soliton_pdf.py - Debug PDF processing step by step
import asyncio
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

async def debug_pdf_processing():
    pdf_path = r'{PROJECT_ROOT}\docs\Living Soliton Memory Systems.pdf'
    
    print("🔍 Step 1: Check if file exists...")
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        return
    print(f"✅ File exists: {Path(pdf_path).stat().st_size / 1024:.2f} KB")
    
    print("\n🔍 Step 2: Extract text from PDF...")
    try:
        from core.universal_file_extractor import extract_content_from_file
        content = await extract_content_from_file(pdf_path)
        print(f"✅ Extracted {len(content)} characters")
        print(f"   Preview: {content[:200]}...")
    except Exception as e:
        print(f"❌ PDF extraction failed: {e}")
        return
    
    print("\n🔍 Step 3: Extract concepts...")
    try:
        from core.concept_extractor_enhanced import ProductionConceptExtractor
        extractor = ProductionConceptExtractor()
        concepts = await extractor.extract_concepts(content)
        print(f"✅ Extracted {len(concepts)} concepts")
        for i, concept in enumerate(concepts[:5]):
            print(f"   - Concept {i+1}: {concept.text[:80]}...")
    except Exception as e:
        print(f"❌ Concept extraction failed: {e}")
        return
    
    print("\n🔍 Step 4: Test embedding service...")
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8080/health")
            if response.status_code == 200:
                print("✅ Embedding service is healthy")
                health = response.json()
                print(f"   - Model: {health.get('model', 'unknown')}")
                print(f"   - Device: {health.get('device', 'unknown')}")
            else:
                print(f"❌ Embedding service returned status {response.status_code}")
                return
    except Exception as e:
        print(f"❌ Cannot connect to embedding service: {e}")
        print("   Make sure to run: python run_embedding_service.py")
        return
    
    print("\n🔍 Step 5: Generate embeddings...")
    try:
        from core.embedding_client_noauth import embed_concepts
        concept_texts = [c.text for c in concepts]
        embeddings = await embed_concepts(concept_texts[:5])  # Test with first 5
        print(f"✅ Generated embeddings: shape {embeddings.shape}")
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ All components working! The issue might be with the full pipeline.")

if __name__ == "__main__":
    asyncio.run(debug_pdf_processing())
