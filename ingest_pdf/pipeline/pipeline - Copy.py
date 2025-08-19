"""
pipeline/pipeline.py

Main pipeline orchestration for PDF ingestion.
Coordinates all modules for complete processing flow.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Local imports
from .config import (
    ENABLE_CONTEXT_EXTRACTION, ENABLE_FREQUENCY_TRACKING, 
    ENABLE_ENTROPY_PRUNING, ENABLE_ENHANCED_MEMORY_STORAGE,
    ENABLE_PARALLEL_PROCESSING, FILE_SIZE_LIMITS, MAX_PARALLEL_WORKERS,
    ENTROPY_CONFIG
)
from .utils import (
    safe_get, safe_divide, safe_percentage, safe_round, 
    sanitize_dict, get_logger
)
from .io import (
    extract_pdf_metadata, preprocess_with_ocr, extract_chunks,
    extract_title_abstract_safe, process_chunks_parallel, detect_section_type
)
from .quality import (
    analyze_concept_purity, reset_frequency_counter, extract_and_boost_concepts
)
from .pruning import apply_entropy_pruning
from .storage import store_concepts_in_soliton, inject_concept_diff

# Setup logger
logger = get_logger(__name__)


def get_dynamic_limits(file_size_mb: float) -> Tuple[int, int]:
    """
    Get dynamic processing limits based on file size.
    
    Args:
        file_size_mb: File size in megabytes
        
    Returns:
        Tuple of (max_chunks, max_concepts)
    """
    for threshold, max_chunks, max_concepts in FILE_SIZE_LIMITS.values():
        if file_size_mb < threshold:
            return max_chunks, max_concepts
    
    # Return xlarge limits if nothing matches
    _, max_chunks, max_concepts = FILE_SIZE_LIMITS['xlarge']
    return max_chunks, max_concepts


def ingest_pdf_clean(pdf_path: str, 
                    doc_id: Optional[str] = None, 
                    extraction_threshold: float = 0.0, 
                    admin_mode: bool = False, 
                    use_ocr: Optional[bool] = None) -> Dict[str, Any]:
    """
    100% BULLETPROOF PDF INGESTION - ENHANCED VERSION
    Now with OCR, academic structure detection, quality metrics, and parallel processing.
    
    Args:
        pdf_path: Path to the PDF file
        doc_id: Optional document ID (defaults to filename stem)
        extraction_threshold: Minimum score threshold for concepts
        admin_mode: Enable admin mode (unlimited concepts)
        use_ocr: Force OCR usage (None = use config)
        
    Returns:
        Dictionary containing extraction results and metadata
    """
    start_time = datetime.now()
    
    # Ensure all variables have safe defaults
    if doc_id is None:
        doc_id = Path(pdf_path).stem
    
    if use_ocr is None:
        from .config import ENABLE_OCR_FALLBACK
        use_ocr = ENABLE_OCR_FALLBACK
    
    # Safe file operations
    try:
        file_size = os.path.getsize(pdf_path)
        file_size_mb = safe_divide(file_size, 1024 * 1024, 0)
    except:
        file_size_mb = 0
    
    # Get dynamic limits
    MAX_CHUNKS, MAX_TOTAL_CONCEPTS = get_dynamic_limits(file_size_mb)
    
    logger.info(f"🛡️ [ENHANCED BULLETPROOF] Ingesting: {Path(pdf_path).name}")
    logger.info(f"File size: {file_size_mb:.1f} MB, Limits: {MAX_CHUNKS} chunks, {MAX_TOTAL_CONCEPTS} concepts")
    
    try:
        # Extract metadata safely
        doc_metadata = extract_pdf_metadata(pdf_path)
        
        # Try OCR preprocessing if enabled
        ocr_text = None
        if use_ocr:
            ocr_text = preprocess_with_ocr(pdf_path)
            if ocr_text:
                doc_metadata['ocr_used'] = True
        
        # Reset frequency counter safely
        if ENABLE_FREQUENCY_TRACKING:
            try:
                reset_frequency_counter()
                logger.info("✅ Frequency counter reset successfully")
            except Exception as e:
                logger.warning(f"⚠️ Frequency counter reset failed: {e}")
                # Force clear any global frequency state
                try:
                    import gc
                    gc.collect()
                    logger.info("🧹 Forced garbage collection to clear state")
                except:
                    pass
        
        # Extract chunks safely
        if ocr_text:
            # Create chunks from OCR text
            chunks = []
            ocr_lines = ocr_text.split('\n')
            chunk_size = 50  # lines per chunk
            for i in range(0, len(ocr_lines), chunk_size):
                chunk_text = '\n'.join(ocr_lines[i:i+chunk_size])
                chunks.append({
                    'text': chunk_text,
                    'index': len(chunks),
                    'section': detect_section_type(chunk_text)
                })
            logger.info(f"📄 Created {len(chunks)} chunks from OCR text")
        else:
            chunks = extract_chunks(pdf_path)
        
        if not chunks:
            logger.warning(f"No chunks extracted from {pdf_path}")
            return _create_empty_response(pdf_path, start_time, admin_mode)
        
        # Enhance chunks with section detection
        for chunk in chunks:
            if isinstance(chunk, dict) and 'section' not in chunk:
                chunk['section'] = detect_section_type(chunk.get('text', ''))
        
        # Extract title and abstract safely
        title_text, abstract_text = "", ""
        if ENABLE_CONTEXT_EXTRACTION:
            title_text, abstract_text = extract_title_abstract_safe(chunks, pdf_path)
        
        # Process chunks
        chunks_to_process = chunks[:MAX_CHUNKS]
        
        extraction_params = {
            'threshold': extraction_threshold,
            'title': title_text,
            'abstract': abstract_text
        }
        
        # Process chunks (parallel or sequential)
        all_extracted_concepts = _process_chunks(chunks_to_process, extraction_params, MAX_TOTAL_CONCEPTS)
        
        if not all_extracted_concepts:
            logger.error("No concepts extracted!")
            return _create_empty_response(pdf_path, start_time, admin_mode, status="no_concepts")
        
        # Count concepts by method
        semantic_count = sum(1 for c in all_extracted_concepts if "universal" in safe_get(c, "method", ""))
        boosted_count = sum(1 for c in all_extracted_concepts if "file_storage_boosted" in safe_get(c, "method", "") or "boost" in safe_get(c, "method", ""))
        
        # Apply purity filtering
        doc_context = {
            'title': title_text,
            'abstract': abstract_text,
            'filename': Path(pdf_path).name
        }
        pure_concepts = analyze_concept_purity(
            all_extracted_concepts, 
            Path(pdf_path).name, 
            title_text, 
            abstract_text, 
            doc_context
        )
        pure_concepts.sort(
            key=lambda x: safe_get(x, 'quality_score', safe_get(x, 'score', 0)), 
            reverse=True
        )
        
        original_pure_count = len(pure_concepts)
        concept_count = len(pure_concepts)
        prune_stats = None
        
        # Apply entropy pruning if enabled
        if ENABLE_ENTROPY_PRUNING and concept_count > 0:
            pure_concepts, prune_stats = apply_entropy_pruning(pure_concepts, admin_mode)
            concept_count = len(pure_concepts)
            logger.info(f"✅ Enhanced entropy pruning: {concept_count} concepts from {original_pure_count}")
        
        # Store concepts if enabled
        if ENABLE_ENHANCED_MEMORY_STORAGE and concept_count > 0:
            _store_concepts_async(pure_concepts, doc_metadata)
        
        # Inject concepts into cognitive interface
        if concept_count > 0:
            inject_concept_diff(pure_concepts, doc_metadata, Path(pdf_path).name)
        
        # Build response
        return _build_response(
            pdf_path, pure_concepts, all_extracted_concepts, 
            chunks, chunks_to_process, semantic_count, boosted_count,
            title_text, abstract_text, doc_metadata, prune_stats,
            original_pure_count, admin_mode, start_time
        )
        
    except Exception as e:
        logger.error(f"❌ PDF ingestion failed: {e}")
        return _create_error_response(pdf_path, str(e), start_time, admin_mode)


def _process_chunks(chunks_to_process: List[Dict], extraction_params: Dict, max_concepts: int) -> List[Dict]:
    """Process chunks either in parallel or sequentially."""
    if ENABLE_PARALLEL_PROCESSING:
        try:
            # Check for existing event loop
            loop = asyncio.get_running_loop()
            # We're in an existing event loop, run parallel processing in the current loop
            logger.info("✅ Running parallel processing in existing event loop")
            # Create an async wrapper to run in the current context
            async def run_in_loop():
                return await process_chunks_parallel(chunks_to_process, extraction_params)
            # Create a task and run it
            task = loop.create_task(run_in_loop())
            # If we're in a sync context called from async, we need nest_asyncio
            try:
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(task)
            except ImportError:
                # Without nest_asyncio, use ThreadPoolExecutor as fallback
                logger.info("🔄 Using ThreadPoolExecutor for parallel processing")
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS or min(4, os.cpu_count() or 1)) as executor:
                    # Submit all chunks to the executor
                    futures = []
                    for i, chunk in enumerate(chunks_to_process):
                        future = executor.submit(
                            extract_and_boost_concepts,
                            chunk.get('text', ''),
                            extraction_params['threshold'],
                            i,
                            chunk.get('section', 'body'),
                            extraction_params['title'],
                            extraction_params['abstract']
                        )
                        futures.append(future)
                    
                    # Collect results
                    all_concepts = []
                    for future in futures:
                        try:
                            concepts = future.result(timeout=30)
                            all_concepts.extend(concepts)
                            if len(all_concepts) >= max_concepts:
                                logger.info(f"Concept limit reached: {len(all_concepts)}")
                                break
                        except Exception as e:
                            logger.error(f"Chunk processing error: {e}")
                    
                    return all_concepts
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(process_chunks_parallel(chunks_to_process, extraction_params))
    else:
        return _process_chunks_sequential(chunks_to_process, extraction_params, max_concepts)


def _process_chunks_sequential(chunks_to_process: List[Dict], extraction_params: Dict, max_concepts: int) -> List[Dict]:
    """Process chunks sequentially."""
    all_extracted_concepts = []
    
    for i, chunk_data in enumerate(chunks_to_process):
        try:
            if isinstance(chunk_data, dict):
                chunk_text = safe_get(chunk_data, "text", "")
                chunk_index = safe_get(chunk_data, "index", i)
                chunk_section = safe_get(chunk_data, "section", "body")
            else:
                chunk_text = str(chunk_data)
                chunk_index = i
                chunk_section = "body"
            
            if not chunk_text:
                continue
            
            # Extract concepts
            enhanced_concepts = extract_and_boost_concepts(
                chunk_text, extraction_params['threshold'], 
                chunk_index, chunk_section, 
                extraction_params['title'], extraction_params['abstract']
            )
            
            all_extracted_concepts.extend(enhanced_concepts)
            
            # Early exit
            if len(all_extracted_concepts) >= max_concepts:
                logger.info(f"Concept limit reached: {len(all_extracted_concepts)}. Stopping.")
                break
                
        except Exception as e:
            logger.error(f"Error processing chunk {i}: {e}")
            continue
    
    return all_extracted_concepts


def _store_concepts_async(concepts: List[Dict], doc_metadata: Dict) -> None:
    """Store concepts asynchronously with proper event loop handling."""
    try:
        # Check for existing event loop
        try:
            loop = asyncio.get_running_loop()
            # Can't use asyncio.run in existing loop
            logger.warning("⚠️ Skipping enhanced memory storage due to existing event loop")
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            asyncio.run(store_concepts_in_soliton(concepts, doc_metadata))
    except Exception as e:
        logger.warning(f"Enhanced memory storage failed: {e}")


def _create_empty_response(pdf_path: str, start_time: datetime, 
                          admin_mode: bool, status: str = "empty") -> Dict[str, Any]:
    """Create response for empty extraction."""
    return {
        "filename": Path(pdf_path).name,
        "concept_count": 0,
        "concepts": [],
        "concept_names": [],
        "status": status,
        "admin_mode": admin_mode,
        "processing_time_seconds": safe_divide((datetime.now() - start_time).total_seconds(), 1, 0.1)
    }


def _create_error_response(pdf_path: str, error: str, start_time: datetime, 
                          admin_mode: bool) -> Dict[str, Any]:
    """Create response for error case."""
    return {
        "filename": Path(pdf_path).name,
        "concept_count": 0,
        "concept_names": [],
        "concepts": [],
        "status": "error",
        "error_message": error,
        "admin_mode": admin_mode,
        "processing_time_seconds": safe_divide((datetime.now() - start_time).total_seconds(), 1, 0.1)
    }


def _build_response(pdf_path: str, pure_concepts: List[Dict], all_extracted_concepts: List[Dict],
                   chunks: List[Dict], chunks_to_process: List[Dict], 
                   semantic_count: int, boosted_count: int,
                   title_text: str, abstract_text: str, doc_metadata: Dict,
                   prune_stats: Optional[Dict], original_pure_count: int,
                   admin_mode: bool, start_time: datetime) -> Dict[str, Any]:
    """Build the complete response dictionary."""
    concept_count = len(pure_concepts)
    total_time = safe_divide((datetime.now() - start_time).total_seconds(), 1, 0.1)
    
    # Calculate scores
    if pure_concepts:
        valid_scores = [safe_get(c, "quality_score", safe_get(c, "score", 0)) for c in pure_concepts]
        valid_scores = [s for s in valid_scores if s is not None]
        avg_score = safe_divide(sum(valid_scores), len(valid_scores), 0) if valid_scores else 0.0
    else:
        avg_score = 0.0
    
    high_conf_count = sum(1 for c in pure_concepts if safe_get(c, "score", 0) > 0.8)
    high_quality_count = sum(1 for c in pure_concepts if safe_get(c, "quality_score", 0) > 0.8)
    
    # Calculate section distribution
    section_distribution = {}
    for c in pure_concepts:
        section = safe_get(safe_get(c, 'metadata', {}), 'section', 'body')
        section_distribution[section] = section_distribution.get(section, 0) + 1
    
    # Build response
    response = {
        "filename": Path(pdf_path).name,
        "concept_count": concept_count,
        "concept_names": [safe_get(c, 'name', '') for c in pure_concepts],
        "concepts": pure_concepts,
        "status": "success" if concept_count > 0 else "no_concepts",
        "purity_based": True,
        "entropy_pruned": ENABLE_ENTROPY_PRUNING and prune_stats is not None,
        "admin_mode": admin_mode,
        "equal_access": True,
        "performance_limited": True,
        "chunks_processed": len(chunks_to_process),
        "chunks_available": len(chunks),
        "semantic_extracted": semantic_count,
        "file_storage_boosted": boosted_count,
        "average_concept_score": safe_round(avg_score),
        "high_confidence_concepts": high_conf_count,
        "high_quality_concepts": high_quality_count,
        "total_extraction_time": safe_round(total_time),
        "domain_distribution": {"general": concept_count},
        "section_distribution": section_distribution,
        "title_found": bool(title_text),
        "abstract_found": bool(abstract_text),
        "ocr_used": safe_get(doc_metadata, 'ocr_used', False),
        "parallel_processing": ENABLE_PARALLEL_PROCESSING,
        "enhanced_memory_storage": ENABLE_ENHANCED_MEMORY_STORAGE,
        "processing_time_seconds": safe_round(total_time),
        "purity_analysis": {
            "raw_concepts": len(all_extracted_concepts),
            "pure_concepts": original_pure_count,
            "final_concepts": concept_count,
            "purity_efficiency_percent": safe_round(safe_percentage(original_pure_count, len(all_extracted_concepts)), 1),
            "diversity_efficiency_percent": safe_round(safe_percentage(concept_count, original_pure_count), 1),
            "top_concepts": [
                {
                    "name": safe_get(c, 'name', ''),
                    "score": safe_round(safe_get(c, 'score', 0)),
                    "quality_score": safe_round(safe_get(c, 'quality_score', 0)),
                    "methods": [safe_get(c, 'method', 'unknown')],
                    "frequency": safe_get(safe_get(c, 'metadata', {}), 'frequency', 1),
                    "section": safe_get(safe_get(c, 'metadata', {}), 'section', 'body'),
                    "purity_decision": "accepted"
                }
                for c in pure_concepts[:10]
            ]
        }
    }
    
    # Add entropy analysis if available
    if prune_stats:
        # Sanitize stats completely
        clean_stats = sanitize_dict(prune_stats)
        
        total = safe_get(clean_stats, "total", 0)
        selected = safe_get(clean_stats, "selected", 0)
        pruned = safe_get(clean_stats, "pruned", 0)
        final_entropy = safe_get(clean_stats, "final_entropy", 0.0)
        avg_similarity = safe_get(clean_stats, "avg_similarity", 0.0)
        
        response["entropy_analysis"] = {
            "enabled": True,
            "admin_mode": admin_mode,
            "total_before_entropy": total,
            "selected_diverse": selected,
            "pruned_similar": pruned,
            "diversity_efficiency_percent": safe_round(safe_percentage(selected, total), 1),
            "final_entropy": safe_round(final_entropy),
            "avg_similarity": safe_round(avg_similarity),
            "by_category": safe_get(clean_stats, "by_category", {}),
            "config": {
                "max_diverse_concepts": "unlimited" if admin_mode else safe_get(ENTROPY_CONFIG, "max_diverse_concepts"),
                "entropy_threshold": safe_get(ENTROPY_CONFIG, "entropy_threshold", 0.0005),
                "similarity_threshold": safe_get(ENTROPY_CONFIG, "similarity_threshold", 0.83),
                "category_aware": safe_get(ENTROPY_CONFIG, "enable_categories", True)
            },
            "performance": {
                "original_pure_concepts": original_pure_count,
                "final_diverse_concepts": concept_count,
                "reduction_ratio": safe_round(safe_divide(original_pure_count - concept_count, original_pure_count, 0))
            }
        }
    else:
        response["entropy_analysis"] = {
            "enabled": False,
            "reason": "entropy_pruning_disabled" if not ENABLE_ENTROPY_PRUNING else "no_concepts_to_prune"
        }
    
    return response


# Re-export for backward compatibility
__all__ = ['ingest_pdf_clean']
