import re
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
import logging

logger = logging.getLogger(__name__)

def extract_concept_blocks(pdf_path: str, min_len: int = 30) -> List[str]:
    """Return list of non-trivial text blocks from PDF or text files"""
    file_path = Path(pdf_path)
    
    # Handle different file types
    if file_path.suffix.lower() == '.pdf':
        # PDF extraction
        try:
            reader = PyPDF2.PdfReader(str(pdf_path))
            blocks: List[str] = []
            for page in reader.pages:
                text = page.extract_text() or ""
                for blk in re.split(r"\n\s*\n", text):
                    blk = blk.strip()
                    if len(blk) >= min_len:
                        blocks.append(blk)
            return blocks
        except Exception as e:
            logger.error(f"Failed to extract from PDF {pdf_path}: {e}")
            return []
    
    elif file_path.suffix.lower() in ['.txt', '.md']:
        # Text file extraction
        try:
            with open(pdf_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            blocks: List[str] = []
            for blk in re.split(r"\n\s*\n", text):
                blk = blk.strip()
                if len(blk) >= min_len:
                    blocks.append(blk)
            return blocks
        except Exception as e:
            logger.error(f"Failed to extract from text file {pdf_path}: {e}")
            return []
    
    else:
        logger.error(f"Unsupported file type: {file_path.suffix}")
        return []


def infer_section_context(chunks: List[str]) -> List[Dict[str, Any]]:
    """Add section context to chunks using safe heuristics"""
    enhanced_chunks = []
    total = len(chunks)
    
    for i, chunk_text in enumerate(chunks):
        text_lower = chunk_text.lower()[:500]  # Only check first 500 chars for performance
        
        # Determine section with safe defaults
        section = "body"  # Safe default
        
        # First chunk often contains abstract
        if i == 0:
            if "abstract" in text_lower[:200]:
                section = "abstract"
            elif text_lower.strip().startswith("introduction") or "1. introduction" in text_lower:
                section = "introduction"
        # Early chunks might be introduction
        elif i <= 1 and ("introduction" in text_lower[:100] or "1. introduction" in text_lower):
            section = "introduction"
        # Last chunks might be conclusion
        elif i >= total - 2:
            if any(term in text_lower for term in ["conclusion", "concluding remarks", "summary", "5. conclusion"]):
                section = "conclusion"
            elif "reference" in text_lower or "bibliography" in text_lower:
                section = "references"
        
        enhanced_chunks.append({
            "text": chunk_text,
            "section": section,
            "index": i,
            "total_chunks": total
        })
    
    return enhanced_chunks


def extract_chunks(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text chunks with section context from PDF or text files"""
    try:
        # Extract based on file type
        raw_chunks = extract_concept_blocks(pdf_path)
        logger.info(f"📄 Extracted {len(raw_chunks)} chunks from {Path(pdf_path).name}")
        
        if not raw_chunks:
            logger.warning(f"No chunks extracted from {pdf_path}")
            return []
        
        # Add section context
        enhanced_chunks = infer_section_context(raw_chunks)
        return enhanced_chunks
        
    except Exception as e:
        logger.error(f"Failed to extract chunks from {pdf_path}: {e}")
        return []
