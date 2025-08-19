# process_pdf_chunked.py - Process large PDFs in chunks
import asyncio
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import fitz  # PyMuPDF

async def process_pdf_in_chunks(pdf_path: str, chunk_size: int = 100):
    """Process a large PDF in chunks of pages"""
    
    print(f"📚 Processing PDF in chunks: {Path(pdf_path).name}")
    
    try:
        # Open PDF
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"📄 Total pages: {total_pages}")
        print(f"📦 Chunk size: {chunk_size} pages")
        
        all_results = []
        total_concepts = 0
        
        # Process in chunks
        for start_page in range(0, total_pages, chunk_size):
            end_page = min(start_page + chunk_size, total_pages)
            
            print(f"\n🔄 Processing pages {start_page+1}-{end_page}...")
            
            # Extract text for this chunk
            chunk_text = ""
            for page_num in range(start_page, end_page):
                chunk_text += doc[page_num].get_text()
            
            if not chunk_text.strip():
                print(f"  ⚠️ No text found in pages {start_page+1}-{end_page}")
                continue
            
            print(f"  📝 Extracted {len(chunk_text)} characters")
            
            # Create temporary file for this chunk
            temp_file = f"temp_chunk_{start_page}_{end_page}.txt"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(chunk_text)
            
            try:
                # Process this chunk
                from core.canonical_ingestion_production_fixed import ingest_file_production
                
                result = await asyncio.wait_for(
                    ingest_file_production(temp_file, metadata={
                        'source_pdf': pdf_path,
                        'pages': f"{start_page+1}-{end_page}",
                        'chunk_of': total_pages
                    }),
                    timeout=60.0  # 1 minute per chunk
                )
                
                if result.success:
                    print(f"  ✅ Extracted {result.concepts_extracted} concepts")
                    total_concepts += result.concepts_extracted
                    all_results.append(result)
                else:
                    print(f"  ❌ Failed: {result.archive_id}")
                    
            except asyncio.TimeoutError:
                print(f"  ⏱️ Timeout processing pages {start_page+1}-{end_page}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
            finally:
                # Clean up temp file
                import os
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        doc.close()
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"  - Total pages processed: {total_pages}")
        print(f"  - Total concepts extracted: {total_concepts}")
        print(f"  - Successful chunks: {len(all_results)}")
        print(f"  - Average concepts per chunk: {total_concepts / max(1, len(all_results)):.1f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def test_first_pages():
    """Just test the first 10 pages"""
    pdf_path = r"{PROJECT_ROOT}\anewapproach.pdf"
    
    print("🧪 Testing with first 10 pages only...")
    
    try:
        doc = fitz.open(pdf_path)
        
        # Extract first 10 pages
        test_content = ""
        for i in range(min(10, len(doc))):
            page_text = doc[i].get_text()
            test_content += page_text
            print(f"  Page {i+1}: {len(page_text)} chars")
        
        doc.close()
        
        # Save as temp file
        with open("test_10_pages.txt", 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        print(f"\n📝 Total test content: {len(test_content)} characters")
        
        # Process it
        from core.canonical_ingestion_production_fixed import ingest_file_production
        result = await ingest_file_production("test_10_pages.txt")
        
        if result.success:
            print(f"\n✅ Test successful!")
            print(f"  - Concepts: {result.concepts_extracted}")
            print(f"  - Quality: {result.quality_metrics.get('overall_quality', 0):.3f}")
        else:
            print(f"\n❌ Test failed: {result.archive_id}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        import os
        if os.path.exists("test_10_pages.txt"):
            os.remove("test_10_pages.txt")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(test_first_pages())
    else:
        pdf_path = r"{PROJECT_ROOT}\anewapproach.pdf"
        asyncio.run(process_pdf_in_chunks(pdf_path, chunk_size=100))
