# process_large_pdf_safe.py - Process with timeouts and chunking
import asyncio
import time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

async def process_large_pdf_safe():
    pdf_path = r"{PROJECT_ROOT}\anewapproach.pdf"
    
    print(f"📚 Processing: anewapproach.pdf")
    print(f"📏 File size: {Path(pdf_path).stat().st_size / (1024*1024):.2f} MB")
    
    # First, let's test if we can extract the PDF
    print("\n🔍 Testing PDF extraction (first 10 pages)...")
    
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"📄 Total pages: {total_pages}")
        
        # Extract first 10 pages as a test
        test_content = ""
        for i in range(min(10, total_pages)):
            test_content += doc[i].get_text()
        
        print(f"✅ Successfully extracted test content: {len(test_content)} chars from first 10 pages")
        doc.close()
        
        # Now try full processing with timeout
        print(f"\n🚀 Processing full document...")
        
        from core.canonical_ingestion_production_fixed import ingest_file_production
        
        # Set a reasonable timeout (5 minutes for 1000 pages)
        try:
            result = await asyncio.wait_for(
                ingest_file_production(pdf_path),
                timeout=300.0  # 5 minutes
            )
            
            if result.success:
                print(f"\n✅ Success! Extracted {result.concepts_extracted} concepts")
                print(f"📊 Quality: {result.quality_metrics.get('overall_quality', 0):.3f}")
            else:
                print(f"\n❌ Failed: {result.archive_id}")
                
        except asyncio.TimeoutError:
            print("\n⏱️ Processing timed out after 5 minutes")
            print("💡 For very large documents, consider:")
            print("   - Splitting into smaller chunks")
            print("   - Processing specific page ranges")
            print("   - Increasing timeout limits")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(process_large_pdf_safe())
