"""
Math Aware Extractor

This module provides utilities for extracting text from PDFs with special
handling for mathematical content to ensure formula integrity.
"""

import os
import re
import hashlib
import logging
import pathlib
import tempfile
import subprocess
import fitz  # PyMuPDF
from typing import Dict, Any, List, Tuple, Optional, Set

logger = logging.getLogger(__name__)

# Constants
MATH_BLOCK_SIZE_BUCKETS = [800, 1200, 1600, 2000, float('inf')]
METRICS = {
    'math_blocks_total': 0,
    'math_blocks_split_total': 0,
    'alt_text_generated_total': 0,
}


def calculate_file_hash(file_path: pathlib.Path) -> str:
    """
    Calculate SHA-256 hash of a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA-256 hash as hex string
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read the file in chunks to efficiently handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()


def extract_text_from_pdf(pdf_path: pathlib.Path, use_mathpix: bool = True) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text from PDF with math formula preservation
    
    Uses a combination of PyMuPDF for text extraction and optionally
    MathPix for formula recognition.
    
    Args:
        pdf_path: Path to PDF file
        use_mathpix: Whether to use MathPix for formula recognition
        
    Returns:
        Tuple of (extracted text, metadata)
    """
    metadata = {
        'math_blocks': 0,
        'math_block_lengths': [],
        'charts': 0,
        'pages': 0,
        'file_hash': calculate_file_hash(pdf_path)
    }
    
    # Extract text with PyMuPDF
    doc = fitz.open(str(pdf_path))
    metadata['pages'] = len(doc)
    
    text_content = []
    math_blocks = []
    chart_blocks = []
    
    # First pass: Extract text and identify potential math regions
    for page_num, page in enumerate(doc):
        # Get text with layout preservation
        text = page.get_text("text")
        text_content.append(text)
        
        # Look for potential mathematical formulas
        math_patterns = [
            r'\$\$[^$]+\$\$',         # Display LaTeX: $$formula$$
            r'\$[^$]+\$',             # Inline LaTeX: $formula$
            r'\\begin\{equation\}.*?\\end\{equation\}',  # LaTeX equations
            r'\\begin\{align\}.*?\\end\{align\}',        # LaTeX align
            r'\\begin\{eqnarray\}.*?\\end\{eqnarray\}',  # LaTeX eqnarray
            r'\\begin\{math\}.*?\\end\{math\}',          # LaTeX math env
            r'\\begin\{displaymath\}.*?\\end\{displaymath\}',  # LaTeX displaymath
        ]
        
        # Identify math blocks
        for pattern in math_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                math_blocks.append(match)
                metadata['math_block_lengths'].append(len(match))
        
        # Look for images that might be charts using PyMuPDF
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Check if this might be a chart (heuristic: contains text)
            # In a real implementation, use a ML model to detect charts
            # For now, we'll just assume every image might be a chart
            chart_blocks.append(f"[CHART: Page {page_num+1}, Image {img_index+1}]")
    
    metadata['math_blocks'] = len(math_blocks)
    metadata['charts'] = len(chart_blocks)
    
    # Update metrics
    METRICS['math_blocks_total'] += len(math_blocks)
    
    # Second pass: If MathPix is available, use it for formula recognition
    if use_mathpix and math_blocks:
        try:
            # This is a placeholder for MathPix API integration
            # In a real implementation, you would call the MathPix API here
            logger.info(f"Would process {len(math_blocks)} math blocks with MathPix")
            
            # For now, we'll just log that we would use MathPix
            for block in math_blocks:
                logger.debug(f"Math block: {block[:30]}...")
        except Exception as e:
            logger.error(f"Error using MathPix: {str(e)}")
    
    # Generate alt text for charts
    if chart_blocks:
        try:
            # This is a placeholder for chart alt text generation
            # In a real implementation, you would use a model like MatplotAlt
            logger.info(f"Would generate alt text for {len(chart_blocks)} charts")
            
            # For now, we'll just log that we would generate alt text
            for block in chart_blocks:
                logger.debug(f"Chart: {block}")
            
            # Update metrics
            METRICS['alt_text_generated_total'] += len(chart_blocks)
        except Exception as e:
            logger.error(f"Error generating alt text: {str(e)}")
    
    # Combine text content
    full_text = "\n\n".join(text_content)
    
    return full_text, metadata


def convert_pdf_to_markdown(pdf_path: pathlib.Path, use_mathpix: bool = True) -> Tuple[pathlib.Path, Dict[str, Any]]:
    """
    Convert PDF to Markdown with math formula preservation
    
    Args:
        pdf_path: Path to PDF file
        use_mathpix: Whether to use MathPix for formula recognition
        
    Returns:
        Tuple of (path to markdown file, metadata)
    """
    logger.info(f"Converting {pdf_path} to markdown with math preservation")
    
    # Extract text from PDF
    text, metadata = extract_text_from_pdf(pdf_path, use_mathpix)
    
    # Create markdown file
    md_path = pdf_path.with_suffix('.md')
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# {pdf_path.stem}\n\n")
        f.write(text)
    
    logger.info(f"Converted {pdf_path} to {md_path} with {metadata['math_blocks']} math blocks")
    
    return md_path, metadata


def chunk_text_preserving_math(text: str, max_len: int = 1800) -> List[Dict[str, Any]]:
    """
    Chunk text while preserving math formulas
    
    This function ensures that LaTeX math formulas are not split across chunks.
    
    Args:
        text: Text to chunk
        max_len: Maximum chunk length
        
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    current_chunk = ""
    current_start = 0
    current_offset = 0
    
    # Define math patterns to preserve
    math_patterns = [
        r'\$\$[^$]+\$\$',         # Display LaTeX: $$formula$$
        r'\$[^$]+\$',             # Inline LaTeX: $formula$
        r'\\begin\{equation\}.*?\\end\{equation\}',  # LaTeX equations
        r'\\begin\{align\}.*?\\end\{align\}',        # LaTeX align
        r'\\begin\{eqnarray\}.*?\\end\{eqnarray\}',  # LaTeX eqnarray
        r'\\begin\{math\}.*?\\end\{math\}',          # LaTeX math env
        r'\\begin\{displaymath\}.*?\\end\{displaymath\}',  # LaTeX displaymath
    ]
    
    # Find all math blocks to protect
    protected_spans = []
    
    for pattern in math_patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            protected_spans.append((match.start(), match.end()))
    
    # Sort spans by start position
    protected_spans.sort()
    
    # Merge overlapping spans
    if protected_spans:
        merged_spans = [protected_spans[0]]
        for span in protected_spans[1:]:
            prev_start, prev_end = merged_spans[-1]
            start, end = span
            
            if start <= prev_end:
                # Overlapping or adjacent spans, merge them
                merged_spans[-1] = (prev_start, max(prev_end, end))
            else:
                # Non-overlapping span, add it
                merged_spans.append(span)
        
        protected_spans = merged_spans
    
    # Split by paragraphs while respecting protected spans
    paragraphs = []
    last_end = 0
    
    # Split text into paragraphs at double newlines
    for match in re.finditer(r'\n\n', text):
        para_end = match.start()
        
        # Check if this split would cut through a math block
        valid_split = True
        for start, end in protected_spans:
            if start < para_end < end:
                valid_split = False
                break
        
        if valid_split:
            paragraphs.append(text[last_end:para_end])
            last_end = match.end()
    
    # Add the last paragraph
    if last_end < len(text):
        paragraphs.append(text[last_end:])
    
    # Chunk paragraphs while respecting max_len
    pos = 0
    
    for para in paragraphs:
        para_len = len(para)
        
        # If adding this paragraph would exceed max_len, create a new chunk
        if current_chunk and len(current_chunk) + para_len > max_len:
            chunks.append({
                "text": current_chunk,
                "start_offset": current_start,
                "end_offset": current_offset,
                "metadata": {
                    "chunk_size": len(current_chunk),
                    "contains_math": any(pattern in current_chunk for pattern in ['$', '\\begin{', '\\end{'])
                }
            })
            
            current_chunk = para
            current_start = pos
            current_offset = pos + para_len
        else:
            # Add to the current chunk
            if current_chunk:
                current_chunk += "\n\n" + para
                current_offset = pos + para_len
            else:
                current_chunk = para
                current_start = pos
                current_offset = pos + para_len
        
        pos += para_len + 2  # +2 for the newlines
    
    # Add the last chunk if there is one
    if current_chunk:
        chunks.append({
            "text": current_chunk,
            "start_offset": current_start,
            "end_offset": current_offset,
            "metadata": {
                "chunk_size": len(current_chunk),
                "contains_math": any(pattern in current_chunk for pattern in ['$', '\\begin{', '\\end{'])
            }
        })
    
    # Verify that no math blocks were split
    math_blocks_split = 0
    for pattern in math_patterns:
        for chunk in chunks:
            chunk_text = chunk["text"]
            # Check for incomplete math blocks
            if re.search(r'\$\$[^$]*$', chunk_text) or re.search(r'^\s*[^$]*\$\$', chunk_text):
                math_blocks_split += 1
            elif re.search(r'\$[^$]*$', chunk_text) or re.search(r'^\s*[^$]*\$', chunk_text):
                math_blocks_split += 1
            elif re.search(r'\\begin\{[^}]*\}[^\\]*$', chunk_text):
                math_blocks_split += 1
            elif re.search(r'^\s*[^\\]*\\end\{[^}]*\}', chunk_text):
                math_blocks_split += 1
    
    # Update metrics
    METRICS['math_blocks_split_total'] += math_blocks_split
    
    if math_blocks_split > 0:
        logger.warning(f"Detected {math_blocks_split} split math blocks")
    
    return chunks


def chunk_markdown_file(md_path: pathlib.Path, max_len: int = 1800) -> List[Dict[str, Any]]:
    """
    Chunk a markdown file while preserving math formulas
    
    Args:
        md_path: Path to markdown file
        max_len: Maximum chunk length
        
    Returns:
        List of chunk dictionaries
    """
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chunks = chunk_text_preserving_math(content, max_len)
    
    # Add source file to metadata
    for chunk in chunks:
        chunk["metadata"]["source_file"] = str(md_path)
    
    # Report chunk size histogram
    sizes = [len(chunk["text"]) for chunk in chunks]
    histogram = [0] * len(MATH_BLOCK_SIZE_BUCKETS)
    
    for size in sizes:
        for i, bucket in enumerate(MATH_BLOCK_SIZE_BUCKETS):
            if size <= bucket:
                histogram[i] += 1
                break
    
    logger.info(f"Chunk size histogram: {histogram}")
    
    return chunks


def extract_images_from_pdf(pdf_path: pathlib.Path, output_dir: Optional[pathlib.Path] = None) -> List[pathlib.Path]:
    """
    Extract images from PDF
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save images (default: same as PDF)
        
    Returns:
        List of paths to extracted images
    """
    if output_dir is None:
        output_dir = pdf_path.parent / f"{pdf_path.stem}_images"
    
    os.makedirs(output_dir, exist_ok=True)
    
    doc = fitz.open(str(pdf_path))
    image_paths = []
    
    for page_num, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Save the image
            image_path = output_dir / f"page{page_num+1}_img{img_index+1}.{image_ext}"
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            image_paths.append(image_path)
    
    return image_paths


def generate_chart_alt_text(image_path: pathlib.Path) -> str:
    """
    Generate alt text for a chart image
    
    This is a placeholder. In a real implementation, you would use
    a model like MatplotAlt or a chart derendering system.
    
    Args:
        image_path: Path to chart image
        
    Returns:
        Generated alt text
    """
    # Placeholder implementation
    return f"[Chart from {image_path.name}]"
