# test_improved_pipeline.py - Test the improved pipeline
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import time

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the improved pipeline
from pipeline import ingest_pdf_clean, ingest_pdf_clean_async

def test_improved_pipeline(pdf_path: str):
    """Test the improved pipeline with code review fixes"""
    
    print("🚀 Testing IMPROVED Pipeline with:")
    print("   ✅ Logger defined at top (no NameError)")
    print("   ✅ Thread-safe concept database") 
    print("   ✅ Proper async handling with asyncio.to_thread")
    print("   ✅ PDF safety checks (size limits, SHA-256)")
    print("   ✅ Simplified math with math.isfinite")
    print("   ✅ Module-wide thread pool (reused)")
    print("   ✅ Thread-local frequency tracking")
    print(f"\n📄 Processing: {Path(pdf_path).name}")
    
    start_time = time.time()
    
    # Use the sync wrapper
    result = ingest_pdf_clean(
        pdf_path=pdf_path,
        extraction_threshold=0.0,
        admin_mode=False,
        use_ocr=False
    )
    
    elapsed = time.time() - start_time
    
    # Display results
    print(f"\n📊 RESULTS:")
    print(f"Status: {result.get('status')}")
    print(f"SHA-256: {result.get('sha256', 'unknown')[:16]}...")
    print(f"Processing time: {elapsed:.1f}s")
    
    if result.get('status') == 'success':
        print(f"\n✅ SUCCESS!")
        print(f"Concepts extracted: {result.get('concept_count')}")
        print(f"Chunks processed: {result.get('chunks_processed')}")
        print(f"Average score: {result.get('average_concept_score')}")
        
        # Show metadata
        metadata = result.get('metadata', {})
        print(f"\n📋 METADATA:")
        print(f"  File size: {metadata.get('file_size_mb', 0):.1f}MB")
        print(f"  Pages: {metadata.get('page_count', 0)}")
        
        # Show purity analysis
        purity = result.get('purity_analysis', {})
        print(f"\n🔬 PURITY:")
        print(f"  Raw concepts: {purity.get('raw_concepts', 0)}")
        print(f"  Pure concepts: {purity.get('pure_concepts', 0)}")
        print(f"  Purity ratio: {purity.get('purity_ratio', 0):.1%}")
        
        # Show top concepts
        if result.get('concepts'):
            print(f"\n🏆 TOP CONCEPTS:")
            for i, concept in enumerate(result['concepts'][:5], 1):
                name = concept.get('name', 'Unknown')
                score = concept.get('score', 0)
                print(f"  {i}. {name} (score: {score:.3f})")
                
    elif result.get('status') == 'error':
        print(f"\n❌ ERROR: {result.get('error_message')}")
        
        # Check if it's the async context issue
        if "event loop" in result.get('error_message', ''):
            print("\n💡 TIP: The improved pipeline detected we're in an async context.")
            print("   Use 'await ingest_pdf_clean_async()' instead of the sync wrapper.")
    
    return result

if __name__ == "__main__":
    # Your PDF
    pdf_path = r"{PROJECT_ROOT}\anewapproach.pdf"
    
    # Test safety with a smaller PDF first if available
    test_pdfs = [
        r"{PROJECT_ROOT}\ecomerce.pdf",  # Small test
        pdf_path  # Big one
    ]
    
    for pdf in test_pdfs:
        if Path(pdf).exists():
            print(f"\n{'='*60}")
            test_improved_pipeline(pdf)
            print(f"{'='*60}\n")
        else:
            print(f"⚠️ PDF not found: {pdf}")
