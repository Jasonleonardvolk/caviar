# process_pdf_robust_fast.py - Robust PDF processing with retry logic
import asyncio
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import fitz
import time
import json

async def process_pdf_robust(pdf_path: str, chunk_size: int = 5):
    """Process PDF with error handling and smaller chunks"""
    
    print(f"🚀 Processing: {Path(pdf_path).name}")
    print(f"⚡ Using smaller chunks (5 pages) for reliability")
    
    start_time = time.time()
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"📄 Total pages: {total_pages}")
    
    successful = 0
    failed = 0
    total_concepts = 0
    failed_chunks = []
    
    for start in range(0, total_pages, chunk_size):
        end = min(start + chunk_size, total_pages)
        
        print(f"\n🔄 Pages {start+1}-{end}...", end="", flush=True)
        
        # Extract text
        text = ""
        for page_num in range(start, end):
            try:
                text += doc[page_num].get_text()
            except Exception as e:
                print(f" [error extracting page {page_num+1}]")
                continue
        
        if not text.strip():
            print(" [empty]")
            continue
        
        # Limit text size more aggressively
        if len(text) > 50000:
            text = text[:50000]
            print(" [truncated]", end="")
        
        # Save chunk
        temp_file = f"temp_chunk_{start}_{end}.txt"
        success = False
        
        for attempt in range(2):  # Try twice
            try:
                with open(temp_file, 'w', encoding='utf-8', errors='ignore') as f:
                    f.write(text)
                
                from core.canonical_ingestion_production_fixed import ingest_file_production
                
                result = await asyncio.wait_for(
                    ingest_file_production(temp_file),
                    timeout=30.0  # 30 second timeout
                )
                
                if result.success:
                    print(f" ✅ {result.concepts_extracted} concepts")
                    successful += 1
                    total_concepts += result.concepts_extracted
                    success = True
                    break
                else:
                    if attempt == 0:
                        print(f" ⚠️ Retry...", end="")
                        await asyncio.sleep(2)
                    else:
                        print(f" ❌ Failed: {result.archive_id}")
                        failed_chunks.append((start+1, end))
                        failed += 1
                        
            except asyncio.TimeoutError:
                print(f" ⏱️ Timeout", end="")
                if attempt == 0:
                    await asyncio.sleep(2)
                else:
                    failed_chunks.append((start+1, end))
                    failed += 1
                    
            except Exception as e:
                print(f" ❌ Error: {str(e)[:30]}")
                if attempt == 0:
                    await asyncio.sleep(2)
                else:
                    failed_chunks.append((start+1, end))
                    failed += 1
                    
            finally:
                import os
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        # Small delay between chunks
        if not success:
            await asyncio.sleep(1)
    
    doc.close()
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"📊 PROCESSING COMPLETE!")
    print(f"  - Total pages: {total_pages}")
    print(f"  - Successful chunks: {successful}")
    print(f"  - Failed chunks: {failed}")
    print(f"  - Total concepts: {total_concepts}")
    print(f"  - Time taken: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"  - Average: {total_concepts/max(1,successful*chunk_size):.1f} concepts/page")
    
    if failed_chunks:
        print(f"\n❌ Failed pages:")
        for start, end in failed_chunks:
            print(f"  - Pages {start}-{end}")
        
        # Save failed chunks info
        with open("failed_chunks.json", "w") as f:
            json.dump({"pdf": pdf_path, "failed_chunks": failed_chunks}, f)
        print(f"\n💡 Failed chunks saved to failed_chunks.json")

if __name__ == "__main__":
    pdf_path = r"{PROJECT_ROOT}\anewapproach.pdf"
    asyncio.run(process_pdf_robust(pdf_path, chunk_size=5))
