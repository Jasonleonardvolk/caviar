# batch_process_files.py - Process multiple files through TORI
import asyncio
import os
from pathlib import Path
from core.canonical_ingestion_production_fixed import ingest_file_production

async def process_directory(directory_path: str, file_extensions: list = None):
    """Process all files in a directory"""
    
    if file_extensions is None:
        file_extensions = ['.pdf', '.txt', '.md', '.docx', '.doc']
    
    directory = Path(directory_path)
    if not directory.exists():
        print(f"❌ Directory not found: {directory_path}")
        return
    
    files = []
    for ext in file_extensions:
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'**/*{ext}'))  # Recursive
    
    print(f"📁 Found {len(files)} files to process")
    
    results = []
    for i, file_path in enumerate(files, 1):
        print(f"\n📄 Processing file {i}/{len(files)}: {file_path.name}")
        
        try:
            result = await ingest_file_production(str(file_path))
            
            if result.success:
                print(f"  ✅ Success: {result.concepts_extracted} concepts extracted")
                print(f"  📊 Quality: {result.quality_metrics.get('overall_quality', 0):.3f}")
                results.append({
                    'file': str(file_path),
                    'success': True,
                    'concepts': result.concepts_extracted,
                    'quality': result.quality_metrics.get('overall_quality', 0)
                })
            else:
                print(f"  ❌ Failed: Check archive {result.archive_id}")
                results.append({
                    'file': str(file_path),
                    'success': False,
                    'error': result.archive_id
                })
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                'file': str(file_path),
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n📊 Processing Summary:")
    print(f"  - Total files: {len(files)}")
    print(f"  - Successful: {sum(1 for r in results if r['success'])}")
    print(f"  - Failed: {sum(1 for r in results if not r['success'])}")
    
    if results:
        avg_quality = sum(r.get('quality', 0) for r in results if r['success']) / max(1, sum(1 for r in results if r['success']))
        print(f"  - Average quality: {avg_quality:.3f}")

async def main():
    # Example usage
    await process_directory("./test_documents", ['.pdf', '.txt'])

if __name__ == "__main__":
    asyncio.run(main())
