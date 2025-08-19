# process_pdf_robust.py - Process with health checks and retries
import asyncio
import time
import httpx
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import fitz

async def check_embedding_service():
    """Check if embedding service is healthy"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8080/health", timeout=5.0)
            if response.status_code == 200:
                return True
    except:
        pass
    return False

async def wait_for_service(max_wait=30):
    """Wait for embedding service to be ready"""
    print("⏳ Waiting for embedding service...")
    start = time.time()
    while time.time() - start < max_wait:
        if await check_embedding_service():
            print("✅ Embedding service is ready")
            return True
        await asyncio.sleep(1)
    print("❌ Embedding service not responding")
    return False

async def process_pdf_with_retry(pdf_path: str, chunk_size: int = 50):
    """Process PDF with smaller chunks and retries"""
    
    print(f"📚 Processing PDF with retries: {Path(pdf_path).name}")
    print(f"📦 Using smaller chunk size: {chunk_size} pages")
    
    # Check service first
    if not await wait_for_service():
        print("❌ Please start the embedding service: python run_embedding_service_graceful.py")
        return
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"📄 Total pages: {total_pages}")
        
        successful_chunks = 0
        total_concepts = 0
        failed_chunks = []
        
        for start_page in range(0, total_pages, chunk_size):
            end_page = min(start_page + chunk_size, total_pages)
            
            print(f"\n🔄 Processing pages {start_page+1}-{end_page}...")
            
            # Check service health before each chunk
            if not await check_embedding_service():
                print("  ⚠️ Embedding service not healthy, waiting...")
                await asyncio.sleep(5)
                if not await check_embedding_service():
                    print("  ❌ Skipping chunk - service down")
                    failed_chunks.append((start_page, end_page))
                    continue
            
            # Extract text
            chunk_text = ""
            for page_num in range(start_page, end_page):
                chunk_text += doc[page_num].get_text()
            
            if not chunk_text.strip():
                print(f"  ⚠️ No text in pages {start_page+1}-{end_page}")
                continue
            
            print(f"  📝 Extracted {len(chunk_text)} characters")
            
            # Save chunk
            temp_file = f"temp_chunk_{start_page}_{end_page}.txt"
            with open(temp_file, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(chunk_text[:500000])  # Limit to 500k chars per chunk
            
            # Process with retries
            success = False
            for attempt in range(3):
                try:
                    from core.canonical_ingestion_production_fixed import ingest_file_production
                    
                    result = await asyncio.wait_for(
                        ingest_file_production(temp_file, metadata={
                            'source_pdf': pdf_path,
                            'pages': f"{start_page+1}-{end_page}",
                            'chunk': f"{start_page//chunk_size + 1}/{(total_pages + chunk_size - 1)//chunk_size}"
                        }),
                        timeout=30.0
                    )
                    
                    if result.success:
                        print(f"  ✅ Success: {result.concepts_extracted} concepts")
                        successful_chunks += 1
                        total_concepts += result.concepts_extracted
                        success = True
                        break
                    else:
                        print(f"  ❌ Attempt {attempt+1} failed: {result.archive_id}")
                        
                except asyncio.TimeoutError:
                    print(f"  ⏱️ Attempt {attempt+1} timed out")
                except Exception as e:
                    print(f"  ❌ Attempt {attempt+1} error: {str(e)[:100]}")
                
                if attempt < 2:
                    print(f"  🔄 Retrying in 5 seconds...")
                    await asyncio.sleep(5)
            
            if not success:
                failed_chunks.append((start_page, end_page))
            
            # Clean up
            import os
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            # Small delay between chunks
            await asyncio.sleep(1)
        
        doc.close()
        
        # Summary
        print(f"\n📊 Final Summary:")
        print(f"  - Total pages: {total_pages}")
        print(f"  - Successful chunks: {successful_chunks}")
        print(f"  - Failed chunks: {len(failed_chunks)}")
        print(f"  - Total concepts extracted: {total_concepts}")
        
        if failed_chunks:
            print(f"\n❌ Failed chunks (pages):")
            for start, end in failed_chunks:
                print(f"  - Pages {start+1}-{end}")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    pdf_path = r"{PROJECT_ROOT}\anewapproach.pdf"
    asyncio.run(process_pdf_with_retry(pdf_path, chunk_size=50))
