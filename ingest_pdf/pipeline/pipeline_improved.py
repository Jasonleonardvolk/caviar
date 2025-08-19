"""
pipeline/pipeline_improved.py

Improved main pipeline orchestration for PDF ingestion.
Separates async and sync paths cleanly for better maintainability.
"""

# ------------------------------------------------------------------
# Imports and logging setup
# ------------------------------------------------------------------
import logging
import sys
import os
from typing import Dict, List, Optional, Tuple, Callable, Any, TypedDict
from dataclasses import dataclass, field
import hashlib
import time
from datetime import datetime
from pathlib import Path
import json
import threading
import asyncio
import contextvars
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import PyPDF2

# Use centralized logging configuration
from .logging_config import get_logger

# Get module-specific logger
logger = get_logger(__name__)

# Configurable constants from environment
MAX_PDF_SIZE_MB = int(os.environ.get('MAX_PDF_SIZE_MB', '100'))
MAX_UNCOMPRESSED_SIZE_MB = int(os.environ.get('MAX_UNCOMPRESSED_SIZE_MB', '500'))

# Note: ENABLE_EMOJI_LOGS is now handled by the logging configuration
# Keep for backward compatibility in log messages
ENABLE_EMOJI_LOGS = os.environ.get('ENABLE_EMOJI_LOGS', 'false').lower() == 'true'

# Import utilities
from .utils import (
    safe_num, safe_divide, safe_multiply, safe_percentage,
    safe_round, safe_get, sanitize_dict
)

# Import configuration
from .config import (
    ENABLE_CONTEXT_EXTRACTION, ENABLE_FREQUENCY_TRACKING, 
    ENABLE_ENTROPY_PRUNING, ENABLE_ENHANCED_MEMORY_STORAGE,
    ENABLE_PARALLEL_PROCESSING, FILE_SIZE_LIMITS, MAX_PARALLEL_WORKERS,
    ENTROPY_CONFIG, ENABLE_OCR_FALLBACK, SECTION_WEIGHTS
)

# Import modules
from .io import (
    extract_pdf_metadata as _extract_pdf_metadata_original,
    preprocess_with_ocr, extract_chunks,
    extract_title_abstract_safe, detect_section_type
)

from .quality import (
    analyze_concept_purity as _analyze_concept_purity_original,
    extract_and_boost_concepts as _extract_and_boost_concepts_original
)

from .pruning import apply_entropy_pruning
from .storage import inject_concept_diff

# Type definitions for better type safety
class ProgressEvent(TypedDict):
    stage: str
    percentage: int
    message: str
    timestamp: float
    details: Optional[Dict[str, Any]]

class ExtractionResult(TypedDict):
    filename: str
    concept_count: int
    concepts: List[Dict[str, Any]]
    concept_names: List[str]
    status: str
    processing_time_seconds: float
    metadata: Dict[str, Any]

# Progress tracking with structured events
class ProgressTracker:
    """Enhanced progress tracker with structured event system."""
    
    def __init__(self, total: int, min_change: float = 1.0, min_seconds: float = 0.0):
        self.total = total
        self.current = 0
        self.last_reported_pct = -1
        self.last_reported_time = 0.0
        self.min_change = min_change
        self.min_seconds = min_seconds
        self._lock = threading.RLock()
        self.events: List[ProgressEvent] = []
        
    def update(self, increment: int = 1) -> Optional[ProgressEvent]:
        """Update progress and return event if significant change."""
        with self._lock:
            self.current += increment
            if self.total <= 0:
                return None
                
            pct = (self.current / self.total) * 100
            current_time = time.time()
            
            # Check thresholds
            pct_change_ok = abs(pct - self.last_reported_pct) >= self.min_change
            time_change_ok = (current_time - self.last_reported_time) >= self.min_seconds
            
            if pct_change_ok and (self.min_seconds == 0 or time_change_ok):
                self.last_reported_pct = pct
                self.last_reported_time = current_time
                
                event: ProgressEvent = {
                    "stage": "progress",
                    "percentage": int(pct),
                    "message": f"Progress: {int(pct)}%",
                    "timestamp": current_time,
                    "details": {
                        "current": self.current,
                        "total": self.total
                    }
                }
                self.events.append(event)
                return event
        return None

# Context-local concept database with caching
_current_db = contextvars.ContextVar("concept_db")

@dataclass
class ConceptDB:
    """Concept database with LRU cache for search optimization."""
    storage: List[Dict] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    names: List[str] = field(default_factory=list)
    
    @lru_cache(maxsize=1024)
    def search_concepts_cached(self, chunk_text_lower: str, max_results: int = 25) -> Tuple[Tuple[str, float], ...]:
        """Cached concept search for repeated queries."""
        results = []
        
        for concept in self.storage[:300]:  # Limit for performance
            if len(results) >= max_results:
                break
            
            name = concept.get("name", "")
            if len(name) < 4:
                continue
            
            if name.lower() in chunk_text_lower:
                base_score = self.scores.get(name, 0.5)
                boost = concept.get("boost_multiplier", 1.2)
                score = min(0.98, base_score * boost)
                results.append((name, score))
        
        return tuple(results)  # Tuple for hashability
    
    def search_concepts(self, chunk_text: str, max_results: int = 25) -> List[Dict]:
        """Search concepts with caching."""
        chunk_lower = chunk_text.lower()
        cached_results = self.search_concepts_cached(chunk_lower, max_results)
        
        return [
            {
                "name": name,
                "score": score,
                "method": "file_storage_boosted",
                "source": {"file_storage_matched": True},
                "metadata": {"category": "general"}
            }
            for name, score in cached_results
        ]

@lru_cache(maxsize=1)
def _load_concept_database() -> ConceptDB:
    """Load concept database with caching to avoid repeated file I/O."""
    logger.info("Loading concept database (cached)...")
    start_time = time.time()
    
    all_concepts = []
    
    # Try multiple loading strategies (same as original)
    # ... (keeping the original loading logic for brevity)
    
    # For this example, using a simplified version
    try:
        import importlib.resources as resources
        data_dir = resources.files('ingest_pdf.data')
        concept_file = data_dir / 'concept_file_storage.json'
        all_concepts = json.loads(concept_file.read_text(encoding='utf-8'))
        logger.info(f"Loaded {len(all_concepts)} concepts")
    except Exception as e:
        logger.warning(f"Failed to load concepts: {e}")
        all_concepts = []
    
    # Build ConceptDB
    names = [c["name"] for c in all_concepts]
    scores = {c["name"]: c.get("priority", 0.5) for c in all_concepts}
    
    elapsed = time.time() - start_time
    logger.info(f"Concept database loaded in {elapsed:.2f}s")
    
    return ConceptDB(storage=all_concepts, scores=scores, names=names)

def get_db() -> ConceptDB:
    """Get concept database with context-local caching."""
    try:
        return _current_db.get()
    except LookupError:
        db = _load_concept_database()
        _current_db.set(db)
        return db

# Core synchronous functions
def extract_pdf_metadata_sync(pdf_path: str) -> Dict[str, Any]:
    """Extract PDF metadata synchronously."""
    metadata = {
        "filename": Path(pdf_path).name,
        "file_path": str(pdf_path),
        "extraction_timestamp": datetime.now().isoformat(),
        "extractor_version": "tori_production_improved"
    }
    
    try:
        # Single read for both size and SHA-256
        with open(pdf_path, "rb") as f:
            content = f.read()
            metadata.update({
                "file_size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest()
            })
    except Exception as e:
        logger.warning(f"Could not extract file info: {e}")
        metadata["file_size_bytes"] = 0
        metadata["sha256"] = "unknown"
    
    # Extract PDF-specific metadata
    if pdf_path.lower().endswith('.pdf'):
        try:
            with open(pdf_path, "rb") as f:
                pdf = PyPDF2.PdfReader(f)
                if pdf.metadata:
                    metadata["pdf_metadata"] = {
                        k.lower().replace('/', ''): str(v)
                        for k, v in pdf.metadata.items() if k and v
                    }
                metadata["page_count"] = len(pdf.pages)
        except Exception as e:
            logger.warning(f"Could not extract PDF metadata: {e}")
            metadata["page_count"] = 1
    
    metadata["file_size_mb"] = round(metadata["file_size_bytes"] / (1024 * 1024), 2)
    
    return metadata

def check_pdf_safety_sync(pdf_path: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Check PDF safety synchronously."""
    metadata = extract_pdf_metadata_sync(pdf_path)
    file_size_mb = metadata.get("file_size_mb", 0)
    
    if file_size_mb > MAX_PDF_SIZE_MB:
        return False, f"PDF too large: {file_size_mb:.1f}MB > {MAX_PDF_SIZE_MB}MB limit", metadata
    
    # Check estimated uncompressed size
    page_count = metadata.get("page_count", 0)
    if page_count > 0:
        est_uncompressed = page_count * 0.5  # ~0.5MB per page estimate
        if est_uncompressed > MAX_UNCOMPRESSED_SIZE_MB:
            return False, f"PDF potentially too large when extracted: ~{est_uncompressed:.0f}MB", metadata
    
    return True, "OK", metadata

def process_chunks_sync(chunks: List[Dict], extraction_params: Dict, 
                       max_concepts: int = None, progress_callback: Optional[Callable] = None) -> List[Dict]:
    """Process chunks synchronously with optional parallel processing."""
    all_concepts = []
    total_chunks = len(chunks)
    
    if ENABLE_PARALLEL_PROCESSING:
        # Use ThreadPoolExecutor for CPU-bound parallel processing
        max_workers = min(MAX_PARALLEL_WORKERS or 4, os.cpu_count() or 1)
        
        def process_chunk(args):
            i, chunk_data = args
            try:
                return extract_and_boost_concepts(
                    chunk_data.get('text', ''),
                    extraction_params['threshold'],
                    i,
                    chunk_data.get('section', 'body'),
                    extraction_params['title'],
                    extraction_params['abstract']
                )
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                return []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_args = list(enumerate(chunks))
            for i, concepts in enumerate(executor.map(process_chunk, chunk_args)):
                if progress_callback:
                    progress = min(90, 40 + int((i / max(1, total_chunks)) * 50))
                    progress_callback("concepts", progress, f"Processing chunk {i+1}/{total_chunks}")
                
                all_concepts.extend(concepts)
                
                if max_concepts and len(all_concepts) >= max_concepts:
                    logger.info(f"Concept limit reached: {len(all_concepts)}")
                    break
    else:
        # Sequential processing
        for i, chunk_data in enumerate(chunks):
            if progress_callback:
                progress = min(90, 40 + int((i / max(1, total_chunks)) * 50))
                progress_callback("concepts", progress, f"Processing chunk {i+1}/{total_chunks}")
            
            try:
                concepts = extract_and_boost_concepts(
                    chunk_data.get('text', ''),
                    extraction_params['threshold'],
                    i,
                    chunk_data.get('section', 'body'),
                    extraction_params['title'],
                    extraction_params['abstract']
                )
                all_concepts.extend(concepts)
                
                if max_concepts and len(all_concepts) >= max_concepts:
                    logger.info(f"Concept limit reached: {len(all_concepts)}")
                    break
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                continue
    
    return all_concepts[:max_concepts] if max_concepts else all_concepts

def extract_and_boost_concepts(chunk: str, threshold: float = 0.0, 
                              chunk_index: int = 0, chunk_section: str = "body", 
                              title_text: str = "", abstract_text: str = "") -> List[Dict[str, Any]]:
    """Extract and boost concepts synchronously."""
    # Use the original function
    concepts = _extract_and_boost_concepts_original(
        chunk, threshold, chunk_index, chunk_section, title_text, abstract_text
    )
    
    # Add database boosting
    db = get_db()
    boosted = db.search_concepts(chunk)
    
    # Combine results
    return concepts + boosted

def ingest_pdf_core(pdf_path: str, 
                   doc_id: Optional[str] = None,
                   extraction_threshold: float = 0.0,
                   admin_mode: bool = False,
                   use_ocr: Optional[bool] = None,
                   progress_callback: Optional[Callable] = None) -> ExtractionResult:
    """
    Core synchronous PDF ingestion logic.
    This is the main processing function that can be wrapped for async usage.
    """
    start_time = datetime.now()
    
    # Initialize
    if doc_id is None:
        doc_id = Path(pdf_path).stem
    
    if use_ocr is None:
        use_ocr = ENABLE_OCR_FALLBACK
    
    # Helper for progress updates
    def update_progress(stage: str, pct: int, msg: str):
        if progress_callback:
            try:
                progress_callback(stage, pct, msg)
            except Exception as e:
                logger.debug(f"Progress callback error: {e}")
    
    update_progress("init", 0, "Starting PDF processing...")
    
    # Safety check
    safe, msg, metadata = check_pdf_safety_sync(pdf_path)
    if not safe:
        logger.error(f"PDF safety check failed: {msg}")
        return {
            "filename": Path(pdf_path).name,
            "concept_count": 0,
            "concepts": [],
            "concept_names": [],
            "status": "error",
            "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
            "metadata": {"error": msg}
        }
    
    file_size_mb = metadata.get("file_size_mb", 0)
    sha256 = metadata.get("sha256", "unknown")
    
    logger.info(f"Processing: {Path(pdf_path).name} ({file_size_mb:.1f}MB, SHA256: {sha256[:8]}...)")
    
    # Get dynamic limits
    from .config import FILE_SIZE_LIMITS
    max_chunks, max_concepts = get_dynamic_limits(file_size_mb)
    
    try:
        # OCR preprocessing if needed
        update_progress("ocr", 10, "Checking OCR requirements...")
        ocr_text = None
        if use_ocr:
            ocr_text = preprocess_with_ocr(pdf_path)
            if ocr_text:
                metadata['ocr_used'] = True
        
        # Extract chunks
        update_progress("chunks", 20, "Extracting text chunks...")
        if ocr_text:
            # Create chunks from OCR text
            chunks = []
            ocr_lines = ocr_text.split('\n')
            chunk_size = 50
            for i in range(0, len(ocr_lines), chunk_size):
                chunk_text = '\n'.join(ocr_lines[i:i+chunk_size])
                chunks.append({
                    'text': chunk_text,
                    'index': len(chunks),
                    'section': detect_section_type(chunk_text)
                })
        else:
            chunks = extract_chunks(pdf_path)
        
        update_progress("chunks", 30, f"Extracted {len(chunks)} text chunks")
        
        if not chunks:
            logger.warning(f"No chunks extracted from {pdf_path}")
            return {
                "filename": Path(pdf_path).name,
                "concept_count": 0,
                "concepts": [],
                "concept_names": [],
                "status": "no_content",
                "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
                "metadata": metadata
            }
        
        # Extract title and abstract
        update_progress("context", 35, "Extracting document context...")
        title_text, abstract_text = "", ""
        if ENABLE_CONTEXT_EXTRACTION:
            title_text, abstract_text = extract_title_abstract_safe(chunks, pdf_path)
        
        # Process chunks
        update_progress("concepts", 40, "Starting concept extraction...")
        chunks_to_process = chunks[:max_chunks]
        
        extraction_params = {
            'threshold': extraction_threshold,
            'title': title_text,
            'abstract': abstract_text
        }
        
        all_concepts = process_chunks_sync(
            chunks_to_process, extraction_params, max_concepts, update_progress
        )
        
        if not all_concepts:
            logger.error("No concepts extracted!")
            return {
                "filename": Path(pdf_path).name,
                "concept_count": 0,
                "concepts": [],
                "concept_names": [],
                "status": "no_concepts",
                "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
                "metadata": metadata
            }
        
        # Analyze purity
        update_progress("analysis", 70, "Analyzing concept purity...")
        doc_context = {
            'title': title_text,
            'abstract': abstract_text,
            'filename': Path(pdf_path).name
        }
        pure_concepts = _analyze_concept_purity_original(
            all_concepts, Path(pdf_path).name, title_text, abstract_text, doc_context
        )
        
        # Apply entropy pruning if enabled
        prune_stats = None
        if ENABLE_ENTROPY_PRUNING and len(pure_concepts) > 0:
            update_progress("pruning", 80, "Applying entropy pruning...")
            pure_concepts, prune_stats = apply_entropy_pruning(pure_concepts, admin_mode)
        
        # Store concepts (sync version)
        if ENABLE_ENHANCED_MEMORY_STORAGE and len(pure_concepts) > 0:
            update_progress("storage", 90, "Storing concepts...")
            # Note: This would need a sync version of store_concepts_in_soliton
            # For now, we'll skip this in the sync path
            logger.info("Concept storage would happen here in async mode")
        
        # Inject concepts
        if len(pure_concepts) > 0:
            inject_concept_diff(pure_concepts, metadata, Path(pdf_path).name)
        
        update_progress("complete", 100, "Processing complete!")
        
        # Build result
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "filename": Path(pdf_path).name,
            "concept_count": len(pure_concepts),
            "concepts": pure_concepts,
            "concept_names": [c.get('name', '') for c in pure_concepts],
            "status": "success",
            "processing_time_seconds": round(processing_time, 2),
            "metadata": {
                **metadata,
                "chunks_processed": len(chunks_to_process),
                "chunks_available": len(chunks),
                "title_found": bool(title_text),
                "abstract_found": bool(abstract_text),
                "entropy_pruned": ENABLE_ENTROPY_PRUNING and prune_stats is not None,
                "admin_mode": admin_mode,
                "prune_stats": prune_stats
            }
        }
        
    except Exception as e:
        logger.error(f"PDF ingestion failed: {e}", exc_info=True)
        return {
            "filename": Path(pdf_path).name,
            "concept_count": 0,
            "concepts": [],
            "concept_names": [],
            "status": "error",
            "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
            "metadata": {"error": str(e)}
        }

def get_dynamic_limits(file_size_mb: float) -> Tuple[int, int]:
    """Get dynamic processing limits based on file size."""
    sorted_limits = sorted(FILE_SIZE_LIMITS.values(), key=lambda x: x[0])
    
    for threshold, max_chunks, max_concepts in sorted_limits:
        if file_size_mb < threshold:
            return max_chunks, max_concepts
    
    # Return xlarge limits if nothing matches
    _, max_chunks, max_concepts = FILE_SIZE_LIMITS.get('xlarge', (float('inf'), 5000, 20000))
    return max_chunks, max_concepts

# Async wrappers
async def ingest_pdf_async(pdf_path: str,
                          doc_id: Optional[str] = None,
                          extraction_threshold: float = 0.0,
                          admin_mode: bool = False,
                          use_ocr: Optional[bool] = None,
                          progress_callback: Optional[Callable] = None) -> ExtractionResult:
    """
    Async wrapper for PDF ingestion.
    Runs the sync core in a thread pool to avoid blocking the event loop.
    """
    # Run the sync function in a thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,  # Use default executor
        ingest_pdf_core,
        pdf_path,
        doc_id,
        extraction_threshold,
        admin_mode,
        use_ocr,
        progress_callback
    )
    
    # Handle any async-only operations here
    if ENABLE_ENHANCED_MEMORY_STORAGE and result["concept_count"] > 0:
        try:
            from .storage import store_concepts_in_soliton
            await store_concepts_in_soliton(result["concepts"], result["metadata"])
        except Exception as e:
            logger.warning(f"Async storage failed: {e}")
    
    return result

def ingest_pdf_clean(pdf_path: str,
                    doc_id: Optional[str] = None,
                    extraction_threshold: float = 0.0,
                    admin_mode: bool = False,
                    use_ocr: Optional[bool] = None,
                    progress_callback: Optional[Callable] = None) -> ExtractionResult:
    """
    Clean synchronous entry point.
    Simply calls the core sync function directly.
    """
    return ingest_pdf_core(
        pdf_path,
        doc_id,
        extraction_threshold,
        admin_mode,
        use_ocr,
        progress_callback
    )

# Legacy compatibility
def handle(file_path: str, **kwargs):
    """Legacy adapter for old pipeline.handle() calls."""
    return ingest_pdf_clean(file_path, **kwargs)

def handle_pdf(file_path: str, **kwargs):
    """Legacy adapter specifically for PDF files."""
    return handle(file_path, **kwargs)

# Preload function
def preload_concept_database():
    """Preload concept database to avoid cold-start latency."""
    logger.info("Preloading concept database...")
    db = get_db()
    logger.info(f"Concept database preloaded: {len(db.storage)} concepts")
    return db

# Exports
__all__ = [
    'ingest_pdf_clean',
    'ingest_pdf_async', 
    'ingest_pdf_core',
    'handle',
    'handle_pdf',
    'get_db',
    'preload_concept_database',
    'ProgressTracker',
    'ProgressEvent',
    'ExtractionResult'
]

logger.info("Pipeline loaded - Improved version with clean async/sync separation")
