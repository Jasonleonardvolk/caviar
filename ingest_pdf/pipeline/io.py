"""
pipeline/io.py

Input/Output operations for PDF processing including chunk extraction,
OCR processing, and parallel chunk processing.
"""

import os
import re
import asyncio
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import PyPDF2

# Local imports
from .config import (
    ENABLE_OCR_FALLBACK, OCR_MAX_PAGES, MAX_PARALLEL_WORKERS,
    ENABLE_PARALLEL_PROCESSING, ACADEMIC_SECTIONS, EXTRACTOR_VERSION
)
from .utils import safe_get, safe_divide, get_logger

# Setup logger
logger = get_logger(__name__)

# === Optional OCR imports ===
try:
    import pytesseract
    from PIL import Image
    import pdf2image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("⚠️ OCR libraries not available. Install pytesseract, pdf2image, and PIL for OCR support.")

# Import dependencies with fallback
try:
    from ..extract_blocks import extract_chunks
    # LEGACY: extractConceptsFromDocument moved to spaCy-based extraction in quality.py
    # from ..extractConceptsFromDocument import extractConceptsFromDocument, get_concept_frequency
except ImportError:
    try:
        from extract_blocks import extract_chunks
        # LEGACY: extractConceptsFromDocument moved to spaCy-based extraction in quality.py
        # from extractConceptsFromDocument import extractConceptsFromDocument, get_concept_frequency
    except ImportError:
        logger.error("❌ Failed to import required modules")
        raise


def extract_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Extract metadata from PDF file with safe defaults.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing file metadata
    """
    metadata = {
        "filename": Path(pdf_path).name,
        "file_path": pdf_path,
        "extraction_timestamp": datetime.now().isoformat(),
        "extractor_version": EXTRACTOR_VERSION
    }
    
    try:
        file_size = os.path.getsize(pdf_path)
        metadata["file_size_bytes"] = file_size
        with open(pdf_path, "rb") as f:
            content = f.read()
            metadata["sha256"] = hashlib.sha256(content).hexdigest()
    except Exception as e:
        logger.warning(f"Could not extract file info: {e}")
        metadata["file_size_bytes"] = 0
        metadata["sha256"] = "unknown"
    
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
    
    return metadata


def preprocess_with_ocr(pdf_path: str, max_pages: Optional[int] = None) -> Optional[str]:
    """
    Try to get better text extraction using OCR if needed.
    
    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to OCR (None = use config)
        
    Returns:
        OCR extracted text or None if not needed/available
    """
    if not OCR_AVAILABLE or not ENABLE_OCR_FALLBACK:
        return None
        
    try:
        # Check if PDF has good text layer
        with open(pdf_path, "rb") as f:
            pdf = PyPDF2.PdfReader(f)
            sample_text = ""
            for i in range(min(3, len(pdf.pages))):
                sample_text += pdf.pages[i].extract_text()
        
        # If text extraction is poor, use OCR
        if len(sample_text.strip()) < 100 or sample_text.count('�') > 10:
            logger.info("📸 Poor text extraction detected, attempting OCR...")
            
            # Convert PDF to images with configurable page limit
            if max_pages is None:
                max_pages = OCR_MAX_PAGES
            last_page = len(pdf.pages) if max_pages is None else min(max_pages, len(pdf.pages))
            
            logger.info(f"📸 OCR will process {last_page} pages (max_pages={max_pages})")
            
            # Stream pages to avoid memory issues
            images = pdf2image.convert_from_path(
                pdf_path, 
                dpi=300, 
                first_page=1, 
                last_page=last_page,
                output_folder=None  # Stream mode
            )
            
            ocr_text = ""
            for i, image in enumerate(images):
                logger.info(f"🔍 OCR processing page {i+1}...")
                page_text = pytesseract.image_to_string(image, lang='eng')
                ocr_text += f"\n--- Page {i+1} ---\n{page_text}"
            
            logger.info(f"✅ OCR completed, extracted {len(ocr_text)} characters")
            return ocr_text
    except Exception as e:
        logger.warning(f"⚠️ OCR preprocessing failed: {e}")
    
    return None


def detect_section_type(chunk_text: str) -> str:
    """
    Detect which section of academic paper this chunk belongs to.
    
    Args:
        chunk_text: Text content of the chunk
        
    Returns:
        Section type (e.g., 'abstract', 'methodology', 'body')
    """
    if not chunk_text:
        return "body"
        
    first_lines = chunk_text[:500].lower()
    
    # Check each section type
    for section_type, markers in ACADEMIC_SECTIONS.items():
        for marker in markers:
            # Look for section headers
            if re.search(r'\b' + re.escape(marker) + r'\b', first_lines):
                return section_type
            # Look for numbered sections
            if re.search(r'^\s*\d+\.?\s*' + re.escape(marker), first_lines, re.MULTILINE):
                return section_type
    
    return "body"


def extract_title_abstract_safe(chunks: List[Any], pdf_path: str) -> Tuple[str, str]:
    """
    Extract title and abstract from chunks with complete safety.
    
    Args:
        chunks: List of text chunks
        pdf_path: Path to PDF (used for fallback title)
        
    Returns:
        Tuple of (title_text, abstract_text)
    """
    title_text = ""
    abstract_text = ""
    
    try:
        if chunks and len(chunks) > 0:
            first_chunk = chunks[0]
            if isinstance(first_chunk, dict):
                first_text = first_chunk.get("text", "")
            else:
                first_text = str(first_chunk)
            
            if first_text:
                lines = [ln.strip() for ln in first_text.splitlines() if ln.strip()]
                if lines:
                    candidate = lines[0]
                    if 10 < len(candidate) < 150 and not candidate.endswith('.'):
                        title_text = candidate
                
                lower_text = first_text.lower()
                if "abstract" in lower_text:
                    try:
                        idx = lower_text.index("abstract")
                        abstract_start = idx + len("abstract")
                        while abstract_start < len(first_text) and first_text[abstract_start] in ": \r\t\n":
                            abstract_start += 1
                        abstract_text = first_text[abstract_start:].strip()
                        
                        intro_pos = abstract_text.lower().find("introduction")
                        if intro_pos > 0:
                            abstract_text = abstract_text[:intro_pos].strip()
                        abstract_text = abstract_text[:1000]
                    except:
                        pass
        
        if not title_text:
            filename = Path(pdf_path).stem
            if len(filename) > 10:
                title_text = filename.replace('_', ' ').replace('-', ' ')
    
    except Exception as e:
        logger.debug(f"Could not extract title/abstract: {e}")
    
    return title_text, abstract_text


async def process_chunks_parallel(chunks: List[Dict], extraction_params: Dict) -> List[Dict]:
    """
    Process chunks in parallel for better performance.
    
    Args:
        chunks: List of chunk dictionaries
        extraction_params: Parameters for concept extraction
        
    Returns:
        List of all extracted concepts
    """
    if not ENABLE_PARALLEL_PROCESSING:
        # Fall back to sequential processing
        all_concepts = []
        for i, chunk in enumerate(chunks):
            concepts = extract_and_boost_concepts(
                chunk.get('text', ''),
                extraction_params['threshold'],
                i,
                chunk.get('section', 'body'),
                extraction_params['title'],
                extraction_params['abstract']
            )
            all_concepts.extend(concepts)
        return all_concepts
    
    # Use configurable max workers
    max_workers = MAX_PARALLEL_WORKERS or min(4, os.cpu_count() or 1)
    
    # Use asyncio.to_thread for better event loop integration
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    async def process_chunk_async(i, chunk_data):
        return await asyncio.to_thread(
            extract_and_boost_concepts,
            chunk_data.get('text', ''),
            extraction_params['threshold'],
            i,
            chunk_data.get('section', 'body'),
            extraction_params['title'],
            extraction_params['abstract']
        )
    
    # Process chunks concurrently with limited concurrency
    semaphore = asyncio.Semaphore(max_workers)
    
    async def process_with_semaphore(i, chunk):
        async with semaphore:
            try:
                return await process_chunk_async(i, chunk)
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                return []
    
    # Create tasks
    tasks = [process_with_semaphore(i, chunk) for i, chunk in enumerate(chunks)]
    chunk_results = await asyncio.gather(*tasks)
    
    # Flatten results
    all_concepts = []
    for concepts in chunk_results:
        all_concepts.extend(concepts)
    
    return all_concepts


# Import concept extraction function for this module
def extract_and_boost_concepts(chunk: str, threshold: float = 0.0, 
                              chunk_index: int = 0, chunk_section: str = "body", 
                              title_text: str = "", abstract_text: str = "") -> List[Dict[str, Any]]:
    """
    Extract and boost concepts from a chunk.
    This is imported from quality module but needed here for parallel processing.
    """
    # This will be properly imported from quality module
    # For now, import directly
    from .quality import extract_and_boost_concepts as _extract_and_boost
    return _extract_and_boost(chunk, threshold, chunk_index, chunk_section, title_text, abstract_text)
