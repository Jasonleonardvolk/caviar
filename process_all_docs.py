# process_all_docs.py - Process all PDFs in docs folder
import asyncio
from pathlib import Path
from core.canonical_ingestion_production_fixed import ingest_file_production

async def process_all_pdfs():
    docs_dir = Path("./docs")
    pdf_files = list(docs_dir.glob("*.pdf"))
    
    print(f"📚 Found {len(pdf_files)} PDFs in docs folder")
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n📄 [{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        
        try:
            result = await ingest_file_production(str(pdf_path))
            
            if result.success:
                print(f"  ✅ Success: {result.concepts_extracted} concepts")
                print(f"  📊 Quality: {result.quality_metrics.get('overall_quality', 0):.3f}")
            else:
                print(f"  ❌ Failed: {result.archive_id}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n✅ Processing complete!")

if __name__ == "__main__":
    asyncio.run(process_all_pdfs())
