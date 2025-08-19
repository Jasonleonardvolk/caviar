# process_pdf_blazing_fast.py - Process PDFs with your new speed!
import asyncio
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import fitz
import time

async def process_pdf_fast(pdf_path: str, chunk_size: int = 10):
    """Process PDF with blazing fast embeddings"""
    
    print(f"🚀 Processing: {Path(pdf_path).name}")
    print(f"⚡ Embedding speed: ~70ms per concept")
    
    start_time = time.time()
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"📄 Total pages: {total_pages}")
    
    successful = 0
    total_concepts = 0
    
    for start in range(0, total_pages, chunk_size):
        end = min(start + chunk_size, total_pages)
        
        print(f"\n🔄 Pages {start+1}-{end}...", end="", flush=True)
        
        # Extract text
        text = ""
        for page_num in range(start, end):
            text += doc[page_num].get_text()
        
        if not text.strip():
            print(" [empty]")
            continue
        
        # Save chunk
        temp_file = f"temp_chunk_{start}_{end}.txt"
        with open(temp_file, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(text[:100000])  # 100K char limit
        
        try:
            from core.canonical_ingestion_production_fixed import ingest_file_production
            
            result = await ingest_file_production(temp_file)
            
            if result.success:
                print(f" ✅ {result.concepts_extracted} concepts")
                successful += 1
                total_concepts += result.concepts_extracted
            else:
                print(f" ❌ Failed")
                
        except Exception as e:
            print(f" ❌ Error: {str(e)[:50]}")
        finally:
            import os
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    doc.close()
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"📊 COMPLETE!")
    print(f"  - Pages processed: {total_pages}")
    print(f"  - Concepts extracted: {total_concepts}")
    print(f"  - Time taken: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"  - Speed: {total_pages/elapsed:.1f} pages/second")
    print(f"  - With old speed, this would have taken: {elapsed*26/60:.1f} minutes!")

if __name__ == "__main__":
    # Process the big PDF!
    pdf_path = r"{PROJECT_ROOT}\anewapproach.pdf"
    asyncio.run(process_pdf_fast(pdf_path))
