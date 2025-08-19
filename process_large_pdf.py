# process_large_pdf.py - Process the 1000-page PDF with monitoring
import asyncio
import time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
from core.canonical_ingestion_production_fixed import ingest_file_production

async def process_large_pdf():
    pdf_path = r"{PROJECT_ROOT}\anewapproach.pdf"
    
    # Check file size
    file_size = Path(pdf_path).stat().st_size / (1024 * 1024)  # MB
    print(f"📚 Processing large PDF: anewapproach.pdf")
    print(f"📏 File size: {file_size:.2f} MB (~1,000 pages)")
    print(f"⏳ This may take a while...\n")
    
    start_time = time.time()
    
    try:
        # Process with progress updates
        print("🔄 Stage 1/4: Extracting text from PDF...")
        result = await ingest_file_production(pdf_path)
        
        if result.success:
            elapsed = time.time() - start_time
            
            print(f"\n✅ Ingestion successful!")
            print(f"  - Concepts extracted: {result.concepts_extracted}")
            print(f"  - Embedding dimensions: {result.embedding_result.get('dimensions', 'unknown')}")
            print(f"  - Processing time: {result.processing_time:.2f}s")
            print(f"  - Total elapsed time: {elapsed:.2f}s")
            
            if result.penrose_verification:
                print(f"\n🔍 Penrose Verification:")
                print(f"  - Status: {result.penrose_verification.status}")
                print(f"  - Pass rate: {result.penrose_verification.metadata.get('pass_rate', 0):.3f}")
                print(f"  - Quality gates:")
                for gate, passed in result.penrose_verification.vector_quality.items():
                    status = "✅" if passed else "❌"
                    print(f"    - {gate}: {status}")
            
            print(f"\n📊 Performance Metrics:")
            print(f"  - Pages/second: {1000 / elapsed:.2f}")
            print(f"  - MB/second: {file_size / elapsed:.2f}")
            print(f"  - Concepts/page: {result.concepts_extracted / 1000:.2f}")
            
            print(f"\n📊 Quality Metrics:")
            for metric, value in result.quality_metrics.items():
                if isinstance(value, float):
                    print(f"  - {metric}: {value:.3f}")
            
            print(f"\n📁 Archive ID: {result.archive_id}")
            print(f"\n🎯 Successfully processed 1,000-page document!")
            
        else:
            print(f"\n❌ Ingestion failed")
            print(f"  - Error archived at: {result.archive_id}")
            
            # Try to get error details
            import json
            error_path = f'./psi_archive/{result.archive_id}.json'
            try:
                with open(error_path, 'r') as f:
                    error_data = json.load(f)
                    print(f"  - Error: {error_data.get('error', 'Unknown')}")
            except:
                pass
                
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Error after {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 TORI Large Document Processing Test")
    print("=" * 50)
    asyncio.run(process_large_pdf())
