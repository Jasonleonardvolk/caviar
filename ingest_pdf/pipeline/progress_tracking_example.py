"""
Example: Using ProgressTracker in PDF Processing

This example shows how to integrate throttled progress reporting
into your PDF processing pipeline.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from your pipeline
from ingest_pdf.pipeline.pipeline import ProgressTracker
from ingest_pdf.pipeline import ingest_pdf_clean

async def process_pdf_batch_with_progress(pdf_files: List[Path]):
    """Process multiple PDFs with progress reporting"""
    
    # Create progress tracker for overall batch
    batch_progress = ProgressTracker(
        total=len(pdf_files),
        min_change=10.0  # Report every 10% for batch
    )
    
    results = []
    
    for i, pdf_file in enumerate(pdf_files):
        logger.info(f"📄 Starting: {pdf_file.name}")
        
        try:
            # Process individual PDF
            result = await process_single_pdf_with_progress(pdf_file)
            results.append(result)
            
            # Update batch progress
            if pct := await batch_progress.update():
                logger.info(f"📊 Batch progress: {pct:.0f}% ({i+1}/{len(pdf_files)} files)")
                
        except Exception as e:
            logger.error(f"❌ Failed to process {pdf_file.name}: {e}")
            results.append({"file": str(pdf_file), "error": str(e)})
    
    logger.info("✅ Batch processing complete!")
    return results


async def process_single_pdf_with_progress(pdf_path: Path) -> Dict:
    """Process a single PDF with chunk-level progress"""
    
    # Simulate getting chunks (in real code, use your extraction method)
    chunk_count = estimate_chunk_count(pdf_path)
    
    # Create progress tracker for this PDF
    pdf_progress = ProgressTracker(
        total=chunk_count,
        min_change=5.0  # Report every 5% for individual PDFs
    )
    
    # Simulate processing chunks
    concepts = []
    for chunk_num in range(chunk_count):
        # Simulate chunk processing
        await asyncio.sleep(0.01)  # Replace with actual processing
        
        # In real code:
        # chunk_concepts = await extract_concepts_from_chunk(chunk)
        # concepts.extend(chunk_concepts)
        
        # Update progress
        if pct := await pdf_progress.update():
            logger.info(f"  📈 {pdf_path.name}: {pct:.0f}% processed")
    
    return {
        "file": str(pdf_path),
        "chunks": chunk_count,
        "concepts": len(concepts),
        "status": "success"
    }


def estimate_chunk_count(pdf_path: Path) -> int:
    """Estimate chunks based on file size"""
    size_mb = pdf_path.stat().st_size / (1024 * 1024)
    
    if size_mb < 1:
        return 10
    elif size_mb < 5:
        return 50
    elif size_mb < 25:
        return 200
    else:
        return 500


class WebSocketProgressReporter:
    """Example: Send progress to WebSocket clients"""
    
    def __init__(self, websocket):
        self.websocket = websocket
        self.progress = None
        
    async def process_with_live_updates(self, pdf_path: Path):
        """Process PDF with real-time progress updates"""
        
        chunks = estimate_chunk_count(pdf_path)
        self.progress = ProgressTracker(total=chunks, min_change=2.0)
        
        # Send initial status
        await self.websocket.send_json({
            "type": "start",
            "file": pdf_path.name,
            "total_chunks": chunks
        })
        
        # Process chunks
        for i in range(chunks):
            # Simulate processing
            await asyncio.sleep(0.01)
            
            # Update progress
            if pct := await self.progress.update():
                await self.websocket.send_json({
                    "type": "progress",
                    "file": pdf_path.name,
                    "percentage": round(pct, 1),
                    "chunks_done": i + 1,
                    "chunks_total": chunks
                })
        
        # Send completion
        await self.websocket.send_json({
            "type": "complete",
            "file": pdf_path.name
        })


def sync_example_with_progress():
    """Example for synchronous contexts"""
    
    files = list(Path("pdfs").glob("*.pdf"))
    progress = ProgressTracker(total=len(files), min_change=5.0)
    
    for i, pdf_file in enumerate(files):
        # Process file (sync)
        # result = process_pdf_sync(pdf_file)
        
        # Update progress (sync version)
        if pct := progress.update_sync():
            print(f"Progress: {pct:.0f}% - Processing {pdf_file.name}")


async def main():
    """Demo the progress tracking"""
    
    # Create some test files
    test_files = [
        Path(f"test_{i}.pdf") for i in range(10)
    ]
    
    # Process with progress
    results = await process_pdf_batch_with_progress(test_files)
    
    # Show results
    logger.info("\n📊 Processing Summary:")
    for result in results:
        if "error" in result:
            logger.error(f"  ❌ {result['file']}: {result['error']}")
        else:
            logger.info(f"  ✅ {result['file']}: {result['chunks']} chunks")


if __name__ == "__main__":
    # Run async example
    asyncio.run(main())
    
    # Or run sync example
    # sync_example_with_progress()
