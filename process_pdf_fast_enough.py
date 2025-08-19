# process_pdf_fast_enough.py - Work with current speeds
import asyncio
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import fitz

async def process_pdf_realistic(pdf_path: str):
    """Process PDF accepting ~1 second per embedding"""
    
    print(f"📚 Processing: {Path(pdf_path).name}")
    print(f"⏱️  Expecting ~1 second per chunk")
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"📄 Total pages: {total_pages}")
    
    # Process 5 pages at a time (smaller chunks = faster)
    pages_per_chunk = 5
    successful = 0
    total_concepts = 0
    
    for start in range(0, total_pages, pages_per_chunk):
        end = min(start + pages_per_chunk, total_pages)
        
        print(f"\n🔄 Processing pages {start+1}-{end}...")
        
        # Extract text
        text = ""
        for page_num in range(start, end):
            text += doc[page_num].get_text()
        
        if not text.strip():
            continue
            
        # Save chunk
        temp_file = f"temp_chunk_{start}_{end}.txt"
        with open(temp_file, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(text[:50000])  # Limit to 50K chars
        
        try:
            from core.canonical_ingestion_production_fixed import ingest_file_production
            
            result = await ingest_file_production(temp_file)
            
            if result.success:
                print(f"  ✅ Success: {result.concepts_extracted} concepts")
                successful += 1
                total_concepts += result.concepts_extracted
            else:
                print(f"  ❌ Failed: {result.archive_id}")
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)[:100]}")
        finally:
            import os
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    doc.close()
    
    print(f"\n📊 Summary:")
    print(f"  - Processed: {successful} chunks")
    print(f"  - Total concepts: {total_concepts}")
    print(f"  - Estimated time: {successful} seconds")

if __name__ == "__main__":
    pdf_path = r"{PROJECT_ROOT}\anewapproach.pdf"
    asyncio.run(process_pdf_realistic(pdf_path))
