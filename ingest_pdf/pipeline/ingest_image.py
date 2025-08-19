from __future__ import annotations

"""
pipeline/ingest_image.py

Image file ingestion handler using OCR.
Supports JPEG, PNG, GIF, WebP, TIFF, BMP formats.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from PIL import Image, ImageOps
import pytesseract
import io

# Import common utilities
from .ingest_common import (
    compute_sha256, ProgressTracker, IngestResult,
    chunk_text, compute_psi_state, safe_round
)

# Import concept extraction
from .quality import extract_and_boost_concepts as extract_concepts
from .quality import analyze_concept_purity
from .pruning import apply_entropy_pruning

# Import holographic integration
from .holographic_bus import get_display_api

logger = logging.getLogger("ingest_image")

# === Configuration ===
SUPPORTED_FORMATS = {
    "image/jpeg": [".jpg", ".jpeg"],
    "image/png": [".png"],
    "image/gif": [".gif"],
    "image/webp": [".webp"],
    "image/tiff": [".tif", ".tiff"],
    "image/bmp": [".bmp"],
}

OCR_LANGUAGES = ["eng", "fra", "deu", "spa", "ita", "por", "chi_sim", "jpn", "kor"]

# === Main Handler ===
async def handle(
    file_path: str,
    doc_id: Optional[str] = None,
    extraction_threshold: float = 0.0,
    admin_mode: bool = False,
    progress_callback: Optional[Callable] = None,
    ocr_lang: str = "eng",
    **kwargs
) -> Dict[str, Any]:
    """
    Handle image files using OCR for text extraction.
    
    Args:
        file_path: Path to image file
        doc_id: Optional document ID
        extraction_threshold: Minimum concept score
        admin_mode: Enable unlimited concepts
        progress_callback: Progress callback function
        ocr_lang: OCR language code (default: "eng")
        
    Returns:
        IngestResult dictionary
    """
    start_time = datetime.now()
    progress = ProgressTracker(progress_callback)
    display = get_display_api()
    file_path = Path(file_path)
    
    await progress.send_progress("init", 0, "Starting image processing...")
    await display.update_progress("init", 0, f"Processing {file_path.name}")
    
    try:
        # Load and analyze image
        image_info = await _analyze_image(file_path, progress)
        
        # Perform OCR
        await progress.send_progress("ocr", 30, f"Extracting text via OCR ({ocr_lang})...")
        ocr_text = await _perform_ocr(file_path, ocr_lang, progress)
        
        if not ocr_text.strip():
            # No text found
            result = IngestResult(
                filename=file_path.name,
                file_path=str(file_path),
                media_type="image",
                status="no_concepts",
                warnings=["No text detected in image"],
                dimensions=image_info["dimensions"],
                color_mode=image_info["color_mode"],
                **image_info["file_info"]
            )
            
            await display.complete({"concept_count": 0, "warning": "No text detected"})
            return result.to_dict()
        
        # Extract concepts from OCR text
        concepts = await _extract_concepts_from_text(
            ocr_text, file_path.name, extraction_threshold,
            admin_mode, progress, display
        )
        
        # Create result
        result = IngestResult(
            filename=file_path.name,
            file_path=str(file_path),
            media_type="image",
            concepts=concepts,
            concept_count=len(concepts),
            concept_names=[c.get("name", "") for c in concepts],
            dimensions=image_info["dimensions"],
            color_mode=image_info["color_mode"],
            processing_time_seconds=(datetime.now() - start_time).total_seconds(),
            average_concept_score=sum(c.get("score", 0) for c in concepts) / len(concepts) if concepts else 0,
            psi_state=compute_psi_state(concepts),
            **image_info["file_info"]
        )
        
        # Show completion in hologram
        await display.complete({
            "concept_count": len(concepts),
            "dimensions": image_info["dimensions"],
            "ocr_text_length": len(ocr_text)
        })
        
        await progress.send_progress("complete", 100, "Image processing complete!")
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Image ingestion failed: {e}", exc_info=True)
        await display.error(str(e))
        
        result = IngestResult(
            filename=file_path.name,
            file_path=str(file_path),
            media_type="image",
            status="error",
            error_message=str(e),
            processing_time_seconds=(datetime.now() - start_time).total_seconds()
        )
        
        return result.to_dict()

# === Image Analysis ===
async def _analyze_image(file_path: Path, progress: ProgressTracker) -> Dict[str, Any]:
    """Analyze image properties"""
    await progress.send_progress("analyze", 10, "Analyzing image...")
    
    # Open image
    with Image.open(file_path) as img:
        # RAM-guard: shrink very large images (>4k px on either axis)
        width, height = img.size
        if max(width, height) > 4096:
            scale = 4096 / max(width, height)
            new_size = (int(width * scale), int(height * scale))
            logger.info(f"Down-scaling {file_path.name} from {img.size} to {new_size} for OCR")
            img = ImageOps.contain(img, new_size)  # Downscale large images to prevent RAM spike
        dimensions = img.size  # (width, height)
        color_mode = img.mode  # RGB, RGBA, L, etc.
        
        # Check if image has transparency
        has_alpha = img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info)
        
        # Get basic stats
        if img.mode in ('RGB', 'RGBA'):
            # Convert to numpy for analysis
            import numpy as np
            img_array = np.array(img)
            
            # Calculate dominant colors (simplified)
            if img.mode == 'RGBA':
                # Remove alpha channel for color analysis
                img_array = img_array[:, :, :3]
            
            # Get average color
            avg_color = img_array.mean(axis=(0, 1))
            
            # Calculate brightness
            brightness = np.mean(avg_color) / 255.0
        else:
            avg_color = None
            brightness = 0.5
    
    # File info
    file_info = {
        "filename": file_path.name,
        "file_path": str(file_path),
        "size_bytes": file_path.stat().st_size,
        "size_mb": file_path.stat().st_size / (1024 * 1024),
        "sha256": compute_sha256(file_path),
    }
    
    return {
        "dimensions": dimensions,
        "color_mode": color_mode,
        "has_alpha": has_alpha,
        "avg_color": avg_color.tolist() if avg_color is not None else None,
        "brightness": safe_round(brightness, 2),
        "file_info": file_info
    }

# === OCR Processing ===
async def _perform_ocr(file_path: Path, language: str, progress: ProgressTracker) -> str:
    """Perform OCR on image"""
    try:
        # Check if Tesseract is available
        pytesseract.get_tesseract_version()
    except Exception:
        logger.warning("Tesseract not found, trying alternative OCR...")
        return await _fallback_ocr(file_path)
    
    # Prepare image for OCR
    with Image.open(file_path) as img:
        # RAM-guard: shrink very large images (>4k px on either axis)
        width, height = img.size
        if max(width, height) > 4096:
            scale = 4096 / max(width, height)
            new_size = (int(width * scale), int(height * scale))
            logger.info(f"Down-scaling {file_path.name} from {img.size} to {new_size} for OCR")
            img = ImageOps.contain(img, new_size)  # Downscale large images to prevent RAM spike
        # Convert to RGB if necessary
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        
        # Enhance image for better OCR (optional)
        # Could add: contrast enhancement, denoising, etc.
        
        await progress.send_progress("ocr", 40, "Running OCR...")
        
        # Perform OCR with configuration
        custom_config = r'--oem 3 --psm 6'  # LSTM engine, uniform block of text
        
        try:
            text = pytesseract.image_to_string(
                img,
                lang=language,
                config=custom_config
            )
        except Exception as e:
            logger.warning(f"OCR with language '{language}' failed: {e}")
            # Fallback to English
            text = pytesseract.image_to_string(img, lang='eng')
        
        # Also get detailed data for future enhancements
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        # Calculate OCR confidence
        confidences = [float(c) for c in data['conf'] if int(c) > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        logger.info(f"OCR completed. Text length: {len(text)}, Avg confidence: {avg_confidence:.1f}%")
        
        await progress.send_progress("ocr", 50, f"OCR complete (confidence: {avg_confidence:.0f}%)")
        
        return text
    
async def _fallback_ocr(file_path: Path) -> str:
    """Fallback OCR using alternative methods"""
    # Could implement:
    # - EasyOCR
    # - PaddleOCR
    # - Cloud OCR APIs (Google Vision, AWS Textract)
    logger.warning("No fallback OCR implemented yet")
    return ""

# === Concept Extraction ===
async def _extract_concepts_from_text(
    text: str,
    filename: str,
    threshold: float,
    admin_mode: bool,
    progress: ProgressTracker,
    display: HolographicDisplayAPI
) -> List[Dict[str, Any]]:
    """Extract concepts from OCR text"""
    
    await progress.send_progress("concepts", 60, "Extracting concepts...")
    
    # Chunk the text
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    
    # Extract concepts from each chunk
    all_concepts = []
    for i, chunk in enumerate(chunks):
        chunk_progress = 60 + int((i / len(chunks)) * 20)
        await progress.send_progress(
            "concepts", chunk_progress,
            f"Processing chunk {i+1}/{len(chunks)}"
        )
        
        # Extract concepts
        concepts = extract_concepts(
            chunk.text, threshold, i,
            "body", "", ""  # No title/abstract for images
        )
        
        # Add to holographic display
        for concept in concepts[:5]:  # Limit to avoid spam
            await display.add_concept(concept)
        
        all_concepts.extend(concepts)
    
    # Apply purity analysis
    await progress.send_progress("analysis", 85, "Analyzing concepts...")
    
    pure_concepts = analyze_concept_purity(
        all_concepts, filename, "", "",
        {"filename": filename, "source": "ocr"}
    )
    
    # Apply entropy pruning if not in admin mode
    if not admin_mode and len(pure_concepts) > 50:
        await progress.send_progress("pruning", 90, "Pruning concepts...")
        pure_concepts, _ = apply_entropy_pruning(pure_concepts, admin_mode)
    
    return pure_concepts

# === Multi-Image Support ===
async def handle_multi_page_tiff(
    file_path: str,
    **kwargs
) -> Dict[str, Any]:
    """Handle multi-page TIFF files"""
    file_path = Path(file_path)
    
    # Extract all pages
    pages = []
    with Image.open(file_path) as img:
        # RAM-guard: shrink very large images (>4k px on either axis)
        width, height = img.size
        if max(width, height) > 4096:
            scale = 4096 / max(width, height)
            new_size = (int(width * scale), int(height * scale))
            logger.info(f"Down-scaling {file_path.name} from {img.size} to {new_size} for OCR")
            img = ImageOps.contain(img, new_size)  # Downscale large images to prevent RAM spike
        for i in range(getattr(img, 'n_frames', 1)):
            img.seek(i)
            # Save each frame to memory
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            pages.append(buffer)
    
    # Process each page
    all_concepts = []
    for i, page_buffer in enumerate(pages):
        # Create temporary file or process in memory
        # For now, simplified:
        logger.info(f"Processing TIFF page {i+1}/{len(pages)}")
        # ... OCR each page ...
    
    # Combine results
    # ... 
    
    logger.warning("Multi-page TIFF support not fully implemented")
    return await handle(str(file_path), **kwargs)

# === Public API ===
async def ingest_image(file_path: str, **kwargs) -> Dict[str, Any]:
    """Public API for image ingestion"""
    return await handle(file_path, **kwargs)

# === Supported Formats ===
def get_supported_formats() -> Dict[str, List[str]]:
    """Get supported image formats"""
    return SUPPORTED_FORMATS.copy()

def is_format_supported(file_path: str) -> bool:
    """Check if image format is supported"""
    ext = Path(file_path).suffix.lower()
    for mime, extensions in SUPPORTED_FORMATS.items():
        if ext in extensions:
            return True
    return False

logger.info("Image ingestion handler loaded with OCR support")


