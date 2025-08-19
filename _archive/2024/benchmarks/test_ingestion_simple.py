# test_ingestion_simple.py - Test with text file
import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set environment to use local embeddings
os.environ["TORI_EMBED_MODE"] = "local"
os.environ["TORI_EMBED_URL"] = "http://localhost:8080"

# Adjust Penrose thresholds for testing
os.environ["TORI_ENTROPY_THRESHOLD"] = "2.0"  # Lower threshold
os.environ["TORI_COSINE_THRESHOLD"] = "0.5"   # Lower threshold

async def test_with_text_file():
    """Test ingestion with a simple text file"""
    
    # Create a test text file
    test_content = """
    This document discusses advanced concepts in dynamical systems theory.
    
    The Koopman operator provides a linear representation of nonlinear dynamics.
    It transforms observables in the state space into a higher-dimensional function space.
    
    Spectral analysis of the Koopman operator reveals eigenvalues and eigenfunctions.
    These spectral properties characterize the system's long-term behavior.
    
    Phase space reconstruction techniques enable analysis of experimental data.
    Using delay embeddings, we can reconstruct attractors from time series.
    
    The Penrose tiling exhibits aperiodic patterns with five-fold symmetry.
    This mathematical structure has applications in quasicrystal physics.
    """
    
    # Write test file
    test_file = "test_document.txt"
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    try:
        from core.canonical_ingestion_production_fixed import ingest_file_production
        
        print("🚀 Testing ingestion pipeline with text file...")
        print(f"📄 Processing {test_file}...")
        
        result = await ingest_file_production(test_file)
        
        if result.success:
            print("\n✅ Ingestion successful!")
            print(f"  - Concepts extracted: {result.concepts_extracted}")
            print(f"  - Embedding dimensions: {result.embedding_result.get('dimensions', 'unknown')}")
            print(f"  - Processing time: {result.processing_time:.2f}s")
            
            if result.penrose_verification:
                print(f"\n🔍 Penrose Verification:")
                print(f"  - Status: {result.penrose_verification.status}")
                print(f"  - Pass rate: {result.penrose_verification.metadata.get('pass_rate', 0):.3f}")
                
                print("\n  Quality gates:")
                for gate, passed in result.penrose_verification.vector_quality.items():
                    print(f"    - {gate}: {'✅' if passed else '❌'}")
            
            print(f"\n📊 Overall quality: {result.quality_metrics.get('overall_quality', 0):.3f}")
            
        else:
            print(f"\n❌ Ingestion failed")
            print(f"  - Error: Check archive at {result.archive_id}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    asyncio.run(test_with_text_file())
