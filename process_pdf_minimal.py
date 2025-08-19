# process_pdf_minimal.py - Minimal PDF processing
import asyncio
import fitz
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

async def process_minimal():
    pdf_path = r"{PROJECT_ROOT}\anewapproach.pdf"
    
    print(f"📚 Minimal processing test: {Path(pdf_path).name}")
    
    # Just test first 5 pages
    doc = fitz.open(pdf_path)
    
    # Extract text from first 5 pages
    text = ""
    for i in range(min(5, len(doc))):
        text += doc[i].get_text()
    
    doc.close()
    
    print(f"📝 Extracted {len(text)} chars from first 5 pages")
    
    # Save as simple text file
    with open("test_5pages.txt", "w", encoding="utf-8") as f:
        f.write(text[:10000])  # Just first 10k chars
    
    # Process it
    from core.canonical_ingestion_production_fixed import ingest_file_production
    
    print("🔄 Processing...")
    result = await ingest_file_production("test_5pages.txt")
    
    if result.success:
        print(f"✅ Success! {result.concepts_extracted} concepts extracted")
        print(f"📊 Quality: {result.quality_metrics.get('overall_quality', 0):.3f}")
    else:
        print(f"❌ Failed: {result.archive_id}")
    
    # Cleanup
    import os
    if os.path.exists("test_5pages.txt"):
        os.remove("test_5pages.txt")

if __name__ == "__main__":
    asyncio.run(process_minimal())
