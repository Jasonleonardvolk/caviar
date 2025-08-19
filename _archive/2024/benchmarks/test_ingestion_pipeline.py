# test_ingestion_pipeline.py - Test the full ingestion pipeline
import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set environment to use local embeddings
os.environ["TORI_EMBED_MODE"] = "local"
os.environ["TORI_EMBED_URL"] = "http://localhost:8080"

async def test_ingestion():
    """Test the complete ingestion pipeline"""
    try:
        from core.canonical_ingestion_production_fixed import ingest_file_production
        
        print("🚀 Testing full ingestion pipeline...")
        print("📄 Processing sample.pdf...")
        
        result = await ingest_file_production('tests/fixtures/sample.pdf')
        
        if result.success:
            print("\n✅ Ingestion successful!")
            print(f"  - Concepts extracted: {result.concepts_extracted}")
            print(f"  - Embedding model: {result.embedding_result.get('model', 'unknown')}")
            print(f"  - Dimensions: {result.embedding_result.get('dimensions', 'unknown')}")
            print(f"  - Processing time: {result.processing_time:.2f}s")
            
            if result.penrose_verification:
                print(f"\n🔍 Penrose Verification:")
                print(f"  - Status: {result.penrose_verification.status}")
                print(f"  - Geometric score: {result.penrose_verification.geometric_score:.3f}")
                print(f"  - Phase coherence: {result.penrose_verification.phase_coherence:.3f}")
                print(f"  - Semantic stability: {result.penrose_verification.semantic_stability:.3f}")
            
            print(f"\n📊 Quality Metrics:")
            for metric, value in result.quality_metrics.items():
                if isinstance(value, float):
                    print(f"  - {metric}: {value:.3f}")
            
            print(f"\n📁 Archive ID: {result.archive_id}")
        else:
            print(f"\n❌ Ingestion failed")
            print(f"  - Archive ID: {result.archive_id}")
            print(f"  - Processing time: {result.processing_time:.2f}s")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ingestion())
