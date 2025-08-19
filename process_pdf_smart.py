# process_pdf_smart.py - Smart PDF processing with better extraction
import asyncio
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import fitz

async def process_pdf_smart(pdf_path: str, pages_per_chunk: int = 10, max_concepts: int = 50):
    """Process PDF with smarter chunking and extraction"""
    
    print(f"📚 Smart processing: {Path(pdf_path).name}")
    print(f"📄 Pages per chunk: {pages_per_chunk}")
    print(f"🎯 Max concepts per chunk: {max_concepts}")
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"📖 Total pages: {total_pages}")
        
        successful = 0
        failed = 0
        total_concepts = 0
        
        # Process in smaller chunks for better concept extraction
        for start in range(0, total_pages, pages_per_chunk):
            end = min(start + pages_per_chunk, total_pages)
            
            print(f"\n🔄 Processing pages {start+1}-{end}...")
            
            # Extract text
            text = ""
            for page_num in range(start, end):
                text += doc[page_num].get_text()
            
            if not text.strip():
                print(f"  ⚠️ No text found")
                continue
                
            print(f"  📝 Extracted {len(text)} characters")
            
            # Save chunk
            temp_file = f"temp_chunk_{start}_{end}.txt"
            with open(temp_file, 'w', encoding='utf-8', errors='ignore') as f:
                # Limit text size to avoid overwhelming the system
                f.write(text[:100000])  # 100K chars max
            
            try:
                from core.canonical_ingestion_production_fixed import ingest_file_production
                
                result = await asyncio.wait_for(
                    ingest_file_production(temp_file, metadata={
                        'source_pdf': pdf_path,
                        'pages': f"{start+1}-{end}",
                        'chunk_info': f"Chunk {start//pages_per_chunk + 1} of {(total_pages + pages_per_chunk - 1)//pages_per_chunk}"
                    }),
                    timeout=60.0
                )
                
                if result.success:
                    print(f"  ✅ Success: {result.concepts_extracted} concepts")
                    print(f"  📊 Quality: {result.quality_metrics.get('overall_quality', 0):.3f}")
                    successful += 1
                    total_concepts += result.concepts_extracted
                else:
                    print(f"  ❌ Failed: {result.archive_id}")
                    failed += 1
                    
            except asyncio.TimeoutError:
                print(f"  ⏱️ Timeout")
                failed += 1
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:100]}")
                failed += 1
            finally:
                import os
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            # Small delay
            await asyncio.sleep(0.5)
        
        doc.close()
        
        print(f"\n📊 Summary:")
        print(f"  - Total pages: {total_pages}")
        print(f"  - Successful chunks: {successful}")
        print(f"  - Failed chunks: {failed}")
        print(f"  - Total concepts: {total_concepts}")
        print(f"  - Concepts per page: {total_concepts/max(1, successful*pages_per_chunk):.1f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    pdf_path = r"{PROJECT_ROOT}\anewapproach.pdf"
    
    # Process with 10 pages per chunk (more manageable)
    asyncio.run(process_pdf_smart(pdf_path, pages_per_chunk=10, max_concepts=50))
