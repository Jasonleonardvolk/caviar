"""
Document handler hooks for the ingest pipeline.
"""
from __future__ import annotations
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

__all__ = ["preprocess_document", "postprocess_document", "extract_text_content"]

def preprocess_document(text: str, meta: Dict[str, Any] | None = None) -> str:
    """
    Real implementation:
      • scrub control characters
      • normalise unicode
      • collapse multiple spaces
    """
    if not text:
        return ""
    text = text.replace("\u00A0", " ")        # non-breaking space → space
    text = re.sub(r"[ \t]{2,}", " ", text)    # collapse runs of space / tab
    return text.strip()

def postprocess_document(chunks: list[str], meta: Dict[str, Any] | None = None) -> list[str]:
    """
    Optionally join very short trailing chunks, remove blanks, etc.
    """
    cleaned: list[str] = [c.strip() for c in chunks if c and c.strip()]
    merged: list[str] = []
    buf = ""
    for chunk in cleaned:
        if len(buf) + len(chunk) < 250:   # heuristic 250-char min
            buf += (" " if buf else "") + chunk
        else:
            if buf:
                merged.append(buf)
            buf = chunk
    if buf:
        merged.append(buf)
    return merged

async def extract_text_content(file_path: Path, progress=None) -> str:
    """
    Extract text content from various file types.
    
    Args:
        file_path: Path to the file
        progress: Optional progress tracker
        
    Returns:
        Extracted text content
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    try:
        if extension in ['.txt', '.text']:
            # Plain text
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
        elif extension == '.pdf':
            # PDF extraction
            try:
                import PyPDF2
                content = ""
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    num_pages = len(pdf_reader.pages)
                    
                    for i, page in enumerate(pdf_reader.pages):
                        if progress:
                            await progress.send_progress(
                                "extract", 
                                20 + (20 * i / num_pages),
                                f"Extracting page {i+1}/{num_pages}"
                            )
                        page_text = page.extract_text()
                        if page_text:
                            content += page_text + "\n\n"
            except ImportError:
                logger.warning("PyPDF2 not installed, trying pdfplumber")
                try:
                    import pdfplumber
                    content = ""
                    with pdfplumber.open(file_path) as pdf:
                        num_pages = len(pdf.pages)
                        for i, page in enumerate(pdf.pages):
                            if progress:
                                await progress.send_progress(
                                    "extract",
                                    20 + (20 * i / num_pages), 
                                    f"Extracting page {i+1}/{num_pages}"
                                )
                            page_text = page.extract_text()
                            if page_text:
                                content += page_text + "\n\n"
                except ImportError:
                    logger.error("No PDF library available (install PyPDF2 or pdfplumber)")
                    return ""
                    
        elif extension in ['.html', '.htm']:
            # HTML extraction
            try:
                from bs4 import BeautifulSoup
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    content = soup.get_text()
            except ImportError:
                # Fallback: basic HTML stripping
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()
                    # Remove script and style blocks
                    html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
                    html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
                    # Remove all tags
                    content = re.sub(r'<[^>]+>', ' ', html_content)
                    
        elif extension in ['.md', '.markdown']:
            # Markdown - just read as text (could enhance with markdown parser)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
        else:
            # Unknown type - try as text
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
        # Apply preprocessing
        content = preprocess_document(content)
        return content
        
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return ""
