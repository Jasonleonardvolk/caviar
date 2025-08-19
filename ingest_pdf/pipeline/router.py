"""
pipeline/router.py

Multi-modal file type router for unified ingestion pipeline.
Dispatches to appropriate handlers based on MIME type.
"""

import mimetypes
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import asyncio

logger = logging.getLogger("multimodal_router")

# Handler imports will be added as we create them
from . import ingest_text_like
from ingest_bus.audio import ingest_audio
from ingest_bus.video import ingest_video  
from . import ingest_image

# === MIME Type Handlers Registry ===
HANDLERS = {
    # Text-like formats (PDF, TXT, HTML, Markdown)
    "application/pdf": ingest_text_like.handle,
    "text/plain": ingest_text_like.handle,
    "text/html": ingest_text_like.handle,
    "text/markdown": ingest_text_like.handle,
    "application/x-markdown": ingest_text_like.handle,
    
    # Audio formats (coming soon)
    "audio/": ingest_audio.handle,  # prefix match - covers mpeg, wav, etc.
    # "audio/mp3": ingest_audio.handle,
    # "audio/wav": ingest_audio.handle,
    # "audio/x-wav": ingest_audio.handle,
    # "audio/mp4": ingest_audio.handle,
    # "audio/aac": ingest_audio.handle,
    
    # Video formats (coming soon)
    "video/": ingest_video.handle,
    # "video/mpeg": ingest_video.handle,
    # "video/quicktime": ingest_video.handle,
    # "video/x-msvideo": ingest_video.handle,
    # "video/webm": ingest_video.handle,
    
    # Image formats (coming soon)
    # "image/jpeg": ingest_image.handle,
    # "image/png": ingest_image.handle,
    # "image/gif": ingest_image.handle,
    # "image/webp": ingest_image.handle,
    # "image/tiff": ingest_image.handle,
    # "image/bmp": ingest_image.handle,
}

# === Extension fallbacks for when MIME detection fails ===
EXTENSION_HANDLERS = {
    ".pdf": ingest_text_like.handle,
    ".txt": ingest_text_like.handle,
    ".html": ingest_text_like.handle,
    ".htm": ingest_text_like.handle,
    ".md": ingest_text_like.handle,
    ".markdown": ingest_text_like.handle,
    
    # Audio extensions
    ".mp3": ingest_audio.handle,
    ".wav": ingest_audio.handle,
    # ".m4a": ingest_audio.handle,
    # ".aac": ingest_audio.handle,
    # ".flac": ingest_audio.handle,
    
    # Video extensions
    ".mp4": ingest_video.handle,
    # ".avi": ingest_video.handle,
    # ".mov": ingest_video.handle,
    ".mkv": ingest_video.handle,
    # ".webm": ingest_video.handle,
    
    # Image extensions
    # ".jpg": ingest_image.handle,
    # ".jpeg": ingest_image.handle,
    # ".png": ingest_image.handle,
    # ".gif": ingest_image.handle,
    # ".webp": ingest_image.handle,
    # ".tif": ingest_image.handle,
    # ".tiff": ingest_image.handle,
    # ".bmp": ingest_image.handle,
}

# === Holographic Event Bus Integration ===
hologram_bus = None

def set_hologram_bus(bus):
    """Set the holographic event bus for visual feedback"""
    global hologram_bus
    hologram_bus = bus

async def publish_hologram_event(event_type: str, data: Dict[str, Any]):
    """Publish event to holographic display if connected"""
    if hologram_bus:
        try:
            await hologram_bus.publish({
                "type": event_type,
                "data": data
            })
        except Exception as e:
            logger.debug(f"Hologram event publish failed: {e}")

# === Main Router Function ===
async def ingest_file(
    file_path: str,
    doc_id: Optional[str] = None,
    admin_mode: bool = False,
    progress_callback: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Route file to appropriate handler based on MIME type.
    
    Args:
        file_path: Path to the file to ingest
        doc_id: Optional document ID
        admin_mode: Enable admin mode for unlimited concepts
        progress_callback: Optional callback for progress updates
        **kwargs: Additional handler-specific arguments
        
    Returns:
        IngestResult dictionary with extracted concepts and metadata
    """
    file_path = Path(file_path)
    
    # Ensure file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Detect MIME type
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if not mime_type:
        logger.warning(f"Could not detect MIME type for {file_path}, falling back to extension")
    
    # Enhanced progress callback that also publishes to hologram
    async def enhanced_progress(stage: str, percent: int, message: str):
        # Call original callback
        if progress_callback:
            try:
                progress_callback(stage, percent, message)
            except Exception as e:
                logger.debug(f"Progress callback error: {e}")
        
        # Publish to hologram bus
        await publish_hologram_event("progress", {
            "stage": stage,
            "percent": percent,
            "message": message,
            "file": file_path.name
        })
    
    # Find handler by MIME type
    handler = None
    if mime_type:
        # Try exact match first
        handler = HANDLERS.get(mime_type)
        
        # Try prefix match for general types
        if not handler:
            for mime_prefix, handler_func in HANDLERS.items():
                if mime_type.startswith(mime_prefix):
                    handler = handler_func
                    break
    
    # Fall back to extension-based routing
    if not handler:
        ext = file_path.suffix.lower()
        handler = EXTENSION_HANDLERS.get(ext)
    
    # No handler found
    if not handler:
        raise ValueError(
            f"Unsupported file type: {file_path.name} "
            f"(MIME: {mime_type or 'unknown'}, ext: {file_path.suffix})"
        )
    
    # Log routing decision
    logger.info(
        f"Routing {file_path.name} to handler "
        f"(MIME: {mime_type or 'unknown'}, handler: {handler.__module__})"
    )
    
    # Publish start event to hologram
    await publish_hologram_event("ingest_start", {
        "file": file_path.name,
        "mime_type": mime_type,
        "size_mb": file_path.stat().st_size / (1024 * 1024)
    })
    
    try:
        # Call the handler
        result = await handler(
            str(file_path),
            doc_id=doc_id,
            admin_mode=admin_mode,
            progress_callback=enhanced_progress,
            **kwargs
        )
        
        # Publish completion event
        await publish_hologram_event("ingest_complete", {
            "file": file_path.name,
            "concept_count": result.get("concept_count", 0),
            "success": True
        })
        
        return result
        
    except Exception as e:
        # Publish error event
        await publish_hologram_event("ingest_error", {
            "file": file_path.name,
            "error": str(e)
        })
        raise

# === Convenience Functions ===
def ingest_file_sync(file_path: str, **kwargs) -> Dict[str, Any]:
    """Synchronous wrapper for ingest_file"""
    return asyncio.run(ingest_file(file_path, **kwargs))

def get_supported_extensions() -> list:
    """Get list of supported file extensions"""
    return sorted(list(EXTENSION_HANDLERS.keys()))

def get_supported_mimes() -> list:
    """Get list of supported MIME types"""
    return sorted(list(HANDLERS.keys()))

# === Phase Planning ===
"""
Implementation Phases:

Phase 1 - Text & PDF (COMPLETE)
- PDF with OCR support
- Plain text files
- HTML documents
- Markdown files

Phase 2 - Images (1 day)
- JPEG/PNG via Tesseract OCR
- Multi-page TIFF support
- HEIC/HEIF support
- Smart language detection

Phase 3 - Audio (1-2 days)
- MP3/WAV via Whisper
- Auto language detection
- VAD for silence removal
- Real-time spectral visualization

Phase 4 - Video (2-3 days)
- Extract audio track â†’ Whisper
- Frame sampling for slide OCR
- Motion detection for key frames
- GPU acceleration support

Phase 5 - Production (1 day)
- Prometheus metrics per type
- Resource usage monitoring
- Queue management
- Horizontal scaling
"""

logger.info("Multi-modal router initialized")
logger.info(f"Supported extensions: {', '.join(get_supported_extensions())}")



