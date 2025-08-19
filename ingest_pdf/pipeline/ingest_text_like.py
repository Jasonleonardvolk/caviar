"""
pipeline/ingest_text_like.py

Text-like file ingestion handler.
Supports PDF, TXT, HTML, and Markdown formats.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import asyncio

# Import common utilities
from .ingest_common import (
    compute_sha256, ProgressTracker, IngestResult,
    chunk_text, compute_psi_state, safe_round
)

# Import processing modules
from .quality import extract_and_boost_concepts as extract_concepts
from .quality import analyze_concept_purity
from .pruning import apply_entropy_pruning

# Import holographic integration
from .holographic_bus import get_display_api

# Import file-specific handlers
from .handlers import extract_text_content

# Import configuration
from .config import (
    FILE_SIZE_LIMITS, CHUNK_SIZE, MAX_CHUNKS_DEFAULT,
    MAX_CONCEPTS_DEFAULT, MIN_CONCEPT_SCORE
)

logger = logging.getLogger("ingest_text_like")

# === Configuration ===
SUPPORTED_FORMATS = {
    "application/pdf": [".pdf"],
    "text/plain": [".txt", ".text"],
    "text/html": [".html", ".htm"],
    "text/markdown": [".md", ".markdown"],
    "application/x-markdown": [".md", ".markdown"],
}

# === Main Handler ===
async def handle(
    file_path: str,
    doc_id: Optional[str] = None,
    extraction_threshold: float = MIN_CONCEPT_SCORE,
    admin_mode: bool = False,
    progress_callback: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Handle text-like files (PDF, TXT, HTML, Markdown).
    
    Args:
        file_path: Path to file
        doc_id: Optional document ID
        extraction_threshold: Minimum concept score
        admin_mode: Enable unlimited concepts
        progress_callback: Progress callback function
        
    Returns:
        IngestResult dictionary
    """
    start_time = datetime.now()
    progress = ProgressTracker(progress_callback)
    display = get_display_api()
    file_path = Path(file_path)
    
    await progress.send_progress("init", 0, "Starting text-like processing...")
    await display.update_progress("init", 0, f"Processing {file_path.name}")
    
    try:
        # Get file info
        file_size = file_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        file_hash = compute_sha256(file_path)
        
        file_info = {
            "file_size": file_size,
            "file_size_mb": safe_round(file_size_mb, 2),
            "file_hash": file_hash,
            "media_type": _get_media_type(file_path),
        }
        
        # Extract text content
        await progress.send_progress("extract", 20, "Extracting text content...")
        text_content = await extract_text_content(file_path, progress)
        
        if not text_content.strip():
            # No text found
            result = IngestResult(
                filename=file_path.name,
                file_path=str(file_path),
                media_type=file_info["media_type"],
                status="no_concepts",
                warnings=["No text content found in file"],
                **file_info
            )
            
            await display.complete({"concept_count": 0, "warning": "No text content"})
            return result.to_dict()
        
        # Get dynamic limits based on file size
        max_chunks, max_concepts = _get_dynamic_limits(file_size_mb)
        if admin_mode:
            max_chunks = float('inf')
            max_concepts = float('inf')
        
        # Create chunks
        await progress.send_progress("chunk", 40, "Creating text chunks...")
        chunks = await chunk_text(
            text_content,
            chunk_size=CHUNK_SIZE,
            overlap=50,
            max_chunks=max_chunks
        )
        
        # Extract concepts
        await progress.send_progress("concepts", 60, "Extracting semantic concepts...")
        all_concepts = []
        
        for i, chunk in enumerate(chunks):
            chunk_progress = 60 + (20 * i / len(chunks))
            await progress.send_progress(
                "concepts",
                chunk_progress,
                f"Processing chunk {i+1}/{len(chunks)}..."
            )
            
            concepts = await extract_concepts(
                chunk["text"],
                extraction_threshold=extraction_threshold,
                max_concepts=max_concepts - len(all_concepts) if not admin_mode else None
            )
            
            for concept in concepts:
                concept["chunk_index"] = i
                concept["position"] = chunk.get("position", 0)
                all_concepts.append(concept)
            
            if not admin_mode and len(all_concepts) >= max_concepts:
                break
        
        # Apply pruning
        await progress.send_progress("prune", 80, "Applying entropy-based pruning...")
        pruned_concepts = await apply_entropy_pruning(all_concepts)
        
        # Analyze concept purity
        await progress.send_progress("purity", 90, "Analyzing concept purity...")
        pure_concepts = [c for c in pruned_concepts if analyze_concept_purity(c)]
        
        # Compute psi states
        psi_states = []
        for concept in pure_concepts[:10]:  # Top 10 concepts
            psi_state = compute_psi_state(concept)
            if psi_state:
                psi_states.append(psi_state)
        
        # Prepare result
        result = IngestResult(
            filename=file_path.name,
            file_path=str(file_path),
            media_type=file_info["media_type"],
            status="success",
            doc_id=doc_id or file_hash[:12],
            chunks_processed=len(chunks),
            chunks_available=len(chunks),
            average_concept_score=sum(c.get("score", 0) for c in pure_concepts) / len(pure_concepts) if pure_concepts else 0,
            admin_mode=admin_mode,
            **file_info
        )
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        result = IngestResult(
            filename=file_path.name,
            file_path=str(file_path),
            media_type=_get_media_type(file_path),
            status="error",
            error=str(e),
            warnings=[f"Processing failed: {type(e).__name__}"]
        )
        
        await display.complete({"error": str(e)})
        return result.to_dict()
    
    finally:
        processing_time = (datetime.now() - start_time).total_seconds()
        await progress.send_progress("complete", 100, f"Completed in {processing_time:.1f}s")

# === Helper Functions ===
def _get_dynamic_limits(file_size_mb: float) -> tuple:
    """Get processing limits based on file size"""
    # Sort by threshold to ensure consistent ordering
    sorted_limits = sorted(FILE_SIZE_LIMITS.values(), key=lambda x: x[0])
    
    for threshold, max_chunks, max_concepts in sorted_limits:
        if file_size_mb < threshold:
            return max_chunks, max_concepts
    
    # Return xlarge limits if nothing matches
    _, max_chunks, max_concepts = FILE_SIZE_LIMITS['xlarge']
    return max_chunks, max_concepts

def _get_media_type(file_path: Path) -> str:
    """Get media type from file extension"""
    ext = file_path.suffix.lower()
    for media_type, extensions in SUPPORTED_FORMATS.items():
        if ext in extensions:
            return media_type
    return "text/plain"  # Default

# === Public API ===
async def ingest_text_like(
    file_path: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Public API for text-like ingestion.
    Alias for the handle function.
    """
    return await handle(file_path, **kwargs)

# Module initialization
logger.info("Text-like ingestion handler loaded (PDF, TXT, HTML, Markdown)")
