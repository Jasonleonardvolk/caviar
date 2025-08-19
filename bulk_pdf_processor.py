"""
Bulk PDF Ingestion Script for TONKA Learning
Processes all PDFs in a directory and adds concepts to the mesh
"""

import os
import sys
from pathlib import Path
import requests
import json
import time
import logging
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API configuration
API_BASE_URL = "http://localhost:8002"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/api/upload"
HEALTH_ENDPOINT = f"{API_BASE_URL}/api/health"

class PDFBulkProcessor:
    """Process multiple PDFs and extract concepts for TONKA"""
    
    def __init__(self, data_dir: str, max_workers: int = 3):
        self.data_dir = Path(data_dir)
        self.max_workers = max_workers
        self.results = []
        self.total_concepts = 0
        
    def find_all_pdfs(self) -> List[Path]:
        """Find all PDF files recursively"""
        pdfs = []
        
        # Get PDFs in root directory
        pdfs.extend(self.data_dir.glob("*.pdf"))
        
        # Get PDFs in subdirectories
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir():
                pdfs.extend(subdir.rglob("*.pdf"))
        
        return sorted(pdfs)
    
    def check_api_health(self) -> bool:
        """Check if API is healthy"""
        try:
            response = requests.get(HEALTH_ENDPOINT, timeout=5)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ API is healthy")
                logger.info(f"   PDF Processing: {data.get('pdf_processing_available', False)}")
                logger.info(f"   TONKA: {data.get('tonka_ready', False)} ({data.get('tonka_concepts_loaded', 0)} concepts)")
                return True
            else:
                logger.error(f"❌ API health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Cannot connect to API: {e}")
            return False
    
    def process_single_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a single PDF file"""
        start_time = time.time()
        relative_path = pdf_path.relative_to(self.data_dir)
        
        try:
            logger.info(f"📄 Processing: {relative_path}")
            
            # Open and upload the file
            with open(pdf_path, 'rb') as f:
                files = {'file': (pdf_path.name, f, 'application/pdf')}
                
                # Add progress tracking ID
                progress_id = f"bulk_{int(time.time())}_{pdf_path.stem}"
                params = {'progress_id': progress_id}
                
                response = requests.post(
                    UPLOAD_ENDPOINT,
                    files=files,
                    params=params,
                    timeout=120  # 2 minute timeout for large files
                )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    document = data.get('document', {})
                    concept_count = document.get('concept_count', 0)
                    processing_time = time.time() - start_time
                    
                    logger.info(f"✅ Success: {relative_path} - {concept_count} concepts in {processing_time:.1f}s")
                    
                    return {
                        'success': True,
                        'path': str(relative_path),
                        'filename': pdf_path.name,
                        'concepts': concept_count,
                        'processing_time': processing_time,
                        'size_mb': pdf_path.stat().st_size / (1024 * 1024),
                        'extraction_method': document.get('extractionMethod', 'unknown')
                    }
                else:
                    error = data.get('error', 'Unknown error')
                    logger.error(f"❌ API error for {relative_path}: {error}")
                    return {
                        'success': False,
                        'path': str(relative_path),
                        'error': error
                    }
            else:
                logger.error(f"❌ HTTP error for {relative_path}: {response.status_code}")
                return {
                    'success': False,
                    'path': str(relative_path),
                    'error': f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"❌ Exception processing {relative_path}: {e}")
            return {
                'success': False,
                'path': str(relative_path),
                'error': str(e)
            }
    
    def process_all_pdfs(self, pdfs: List[Path]) -> None:
        """Process all PDFs with parallel execution"""
        total = len(pdfs)
        logger.info(f"🚀 Starting bulk processing of {total} PDFs with {self.max_workers} workers")
        
        processed = 0
        successful = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_pdf = {
                executor.submit(self.process_single_pdf, pdf): pdf 
                for pdf in pdfs
            }
            
            # Process as they complete
            for future in as_completed(future_to_pdf):
                pdf = future_to_pdf[future]
                processed += 1
                
                try:
                    result = future.result()
                    self.results.append(result)
                    
                    if result['success']:
                        successful += 1
                        self.total_concepts += result.get('concepts', 0)
                    else:
                        failed += 1
                    
                    # Progress update
                    progress = (processed / total) * 100
                    logger.info(f"📊 Progress: {processed}/{total} ({progress:.1f}%) - Success: {successful}, Failed: {failed}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process {pdf}: {e}")
                    failed += 1
        
        logger.info(f"🏁 Processing complete!")
        logger.info(f"   Total PDFs: {total}")
        logger.info(f"   Successful: {successful}")
        logger.info(f"   Failed: {failed}")
        logger.info(f"   Total concepts extracted: {self.total_concepts}")
    
    def save_results(self, output_file: str = "pdf_processing_results.json"):
        """Save processing results to file"""
        summary = {
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            'data_directory': str(self.data_dir),
            'total_pdfs': len(self.results),
            'successful': sum(1 for r in self.results if r['success']),
            'failed': sum(1 for r in self.results if not r['success']),
            'total_concepts': self.total_concepts,
            'results': self.results
        }
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"💾 Results saved to: {output_path}")
        
        # Print summary of failures
        failures = [r for r in self.results if not r['success']]
        if failures:
            logger.warning(f"\n⚠️ Failed PDFs ({len(failures)}):")
            for f in failures[:10]:  # Show first 10
                logger.warning(f"   - {f['path']}: {f.get('error', 'Unknown error')}")
            if len(failures) > 10:
                logger.warning(f"   ... and {len(failures) - 10} more")

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bulk PDF processor for TONKA learning")
    parser.add_argument(
        "--data-dir",
        default="C:/Users/jason/Desktop/data",
        help="Directory containing PDFs"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of PDFs to process"
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Only process PDFs matching pattern (e.g. 'arxiv*')"
    )
    
    args = parser.parse_args()
    
    # Create processor
    processor = PDFBulkProcessor(args.data_dir, max_workers=args.workers)
    
    # Check API health
    if not processor.check_api_health():
        logger.error("❌ API is not healthy. Please start the server with: python enhanced_launcher.py")
        return 1
    
    # Find PDFs
    logger.info(f"🔍 Searching for PDFs in: {args.data_dir}")
    all_pdfs = processor.find_all_pdfs()
    
    # Apply pattern filter if specified
    if args.pattern:
        import fnmatch
        all_pdfs = [p for p in all_pdfs if fnmatch.fnmatch(p.name, args.pattern)]
        logger.info(f"🔍 Filtered to {len(all_pdfs)} PDFs matching pattern: {args.pattern}")
    
    # Apply limit if specified
    if args.limit:
        all_pdfs = all_pdfs[:args.limit]
        logger.info(f"🔍 Limited to first {args.limit} PDFs")
    
    if not all_pdfs:
        logger.warning("⚠️ No PDFs found to process")
        return 1
    
    logger.info(f"📚 Found {len(all_pdfs)} PDFs to process")
    
    # Show first few PDFs
    for pdf in all_pdfs[:5]:
        logger.info(f"   - {pdf.relative_to(processor.data_dir)}")
    if len(all_pdfs) > 5:
        logger.info(f"   ... and {len(all_pdfs) - 5} more")
    
    # Confirm before processing
    response = input(f"\n🤔 Process {len(all_pdfs)} PDFs? (y/n): ")
    if response.lower() != 'y':
        logger.info("❌ Cancelled by user")
        return 0
    
    # Process all PDFs
    start_time = time.time()
    processor.process_all_pdfs(all_pdfs)
    total_time = time.time() - start_time
    
    # Save results
    processor.save_results()
    
    # Final summary
    logger.info(f"\n📊 FINAL SUMMARY:")
    logger.info(f"   Total time: {total_time:.1f} seconds")
    logger.info(f"   Average time per PDF: {total_time/len(all_pdfs):.1f} seconds")
    logger.info(f"   Total concepts extracted: {processor.total_concepts}")
    logger.info(f"   Average concepts per PDF: {processor.total_concepts/len(all_pdfs):.1f}")
    
    logger.info(f"\n🎉 TONKA now has {processor.total_concepts} new concepts to learn from!")
    logger.info(f"💡 Restart TONKA to use the updated concept mesh")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
