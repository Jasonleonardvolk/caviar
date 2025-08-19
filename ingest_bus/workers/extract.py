"""
Extract worker module for TORI Ingest Bus with enhanced logging and metadata.

This module provides the functions for extracting, chunking, vectorizing,
and concept-mapping content from different document types. It has been
enhanced to address Issues #1-#4 from the triage document:

- Issue #1: Enhanced logging with LoopRecord entries
- Issue #2: Configurable thresholds with fallback logic
- Issue #3: Removed hard concept caps
- Issue #4: Full metadata preservation
"""

import os
import sys
import time
import json
import hashlib
import logging
import asyncio
import subprocess
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path

# Import data models
from models.schemas import (
    IngestStatus, DocumentType, FailureCode,
    Chunk, ConceptVectorLink, IngestJob
)

# Import enhanced ingest_pdf modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ingest_pdf.threshold_config import (
        MIN_CONFIDENCE, FALLBACK_MIN_COUNT, MAX_CONCEPTS_DEFAULT,
        get_threshold_for_media_type, get_adaptive_threshold, get_fallback_count
    )
    from ingest_pdf.concept_logger import log_loop_record, log_concept_summary, warn_empty_segment
    from ingest_pdf.scoring import filter_concepts, apply_confidence_fallback
    from ingest_pdf.pipeline_validator import validate_concepts
    from ingest_pdf.cognitive_interface import add_concept_diff
    ENHANCED_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Enhanced modules not available: {e}")
    ENHANCED_MODULES_AVAILABLE = False
    # Fallback constants
    MIN_CONFIDENCE = 0.5
    FALLBACK_MIN_COUNT = 3
    MAX_CONCEPTS_DEFAULT = 20

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest-bus.extract")

# Load configuration from parent directory
try:
    config_path = Path(__file__).parent.parent.parent / "conversation_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
except Exception as e:
    logger.warning(f"Could not load configuration: {str(e)}")
    logger.warning("Using default configuration")
    config = {
        "scholar_sphere": {
            "enabled": True,
            "encoder_version": "v2.5.0",
            "chunk_size": 512,
            "chunk_overlap": 128
        }
    }

# ScholarSphere configuration
SCHOLAR_SPHERE_ENABLED = config.get("scholar_sphere", {}).get("enabled", True)
ENCODER_VERSION = config.get("scholar_sphere", {}).get("encoder_version", "v2.5.0")
CHUNK_SIZE = config.get("scholar_sphere", {}).get("chunk_size", 512)
CHUNK_OVERLAP = config.get("scholar_sphere", {}).get("chunk_overlap", 128)

def determine_media_type(file_path: Optional[str], file_content: Optional[bytes]) -> str:
    """Determine media type for threshold configuration."""
    if file_path:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return "pdf"
        elif ext in [".mp3", ".wav", ".m4a"]:
            return "audio"
        elif ext in [".mp4", ".avi", ".mkv"]:
            return "video"
        elif ext in [".json", ".md"] and "conversation" in str(file_path).lower():
            return "conversation"
    
    if file_content:
        # Simple heuristics based on content
        try:
            text_content = file_content.decode('utf-8')[:200]
            if text_content.strip().startswith('{'):
                return "conversation"
        except:
            pass
    
    return "pdf"  # Default

# Extract content from a PDF document
async def extract_pdf(file_path: Optional[str], file_content: Optional[bytes], job: IngestJob) -> Optional[str]:
    """
    Extract text content from a PDF document with enhanced logging.
    
    Args:
        file_path: Path to the PDF file
        file_content: Raw PDF content if file_path is None
        job: The ingest job
        
    Returns:
        str: Extracted text content, or None if extraction failed
    """
    logger.info(f"[LoopRecord] PDF extraction started: job_id={job.id}")
    
    try:
        # Use PyPDF2 if available
        try:
            from PyPDF2 import PdfReader
            
            if file_path:
                reader = PdfReader(file_path)
            else:
                from io import BytesIO
                reader = PdfReader(BytesIO(file_content))
            
            text = ""
            page_count = len(reader.pages)
            
            # Extract text page by page with logging
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += page_text
                
                # Log each page extraction (Issue #1 - LoopRecord logging)
                segment_id = f"{job.id}_pdf_page_{page_num + 1}"
                if ENHANCED_MODULES_AVAILABLE:
                    if page_text.strip():
                        log_loop_record(segment_id, [{"page": page_num + 1, "chars": len(page_text)}])
                    else:
                        warn_empty_segment(segment_id, "PDF page contained no extractable text")
                else:
                    logger.info(f"[LoopRecord] {segment_id}: {len(page_text)} characters extracted")
            
            logger.info(f"[Summary] PDF extraction complete: {page_count} pages, {len(text)} total characters")
            return text
            
        except ImportError:
            logger.warning("PyPDF2 not available, falling back to pdf_reader.py")
            
            # Fall back to custom PDF reader if available
            pdf_reader_path = Path(__file__).parent.parent.parent / "pdf_reader.py"
            if os.path.exists(pdf_reader_path):
                sys.path.append(str(pdf_reader_path.parent))
                from pdf_reader import extract_text_from_pdf
                
                if file_path:
                    return extract_text_from_pdf(file_path)
                else:
                    temp_path = Path("temp") / f"temp_{job.id}.pdf"
                    os.makedirs(temp_path.parent, exist_ok=True)
                    
                    with open(temp_path, "wb") as f:
                        f.write(file_content)
                    
                    result = extract_text_from_pdf(str(temp_path))
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    return result
            else:
                logger.error("No PDF extraction method available")
                return None
        
    except Exception as e:
        logger.exception(f"Error extracting PDF content: {str(e)}")
        return None

# Extract content from a conversation file
async def extract_conversation(file_path: Optional[str], file_content: Optional[bytes], job: IngestJob) -> Optional[str]:
    """
    Extract content from a conversation file (JSON or Markdown) with enhanced logging.
    
    Args:
        file_path: Path to the conversation file
        file_content: Raw conversation content if file_path is None
        job: The ingest job
        
    Returns:
        str: Extracted and processed conversation content, or None if extraction failed
    """
    logger.info(f"[LoopRecord] Conversation extraction started: job_id={job.id}")
    
    try:
        # Determine file path or create a temporary file
        actual_file_path = file_path
        temp_file = None
        
        if not actual_file_path and file_content:
            # Create temporary file
            file_ext = ".json" if file_content.strip().startswith(b"{") else ".md"
            temp_file = Path("temp") / f"temp_{job.id}{file_ext}"
            os.makedirs(temp_file.parent, exist_ok=True)
            
            with open(temp_file, "wb") as f:
                f.write(file_content)
            
            actual_file_path = str(temp_file)
        
        if not actual_file_path:
            logger.error("No file path or content provided for conversation extraction")
            return None
        
        # Check if extract_conversation.js exists
        extractor_path = Path(__file__).parent.parent.parent / "extract_conversation.js"
        if not os.path.exists(extractor_path):
            logger.error(f"Conversation extractor not found at {extractor_path}")
            return None
        
        # Use our custom output directory
        output_dir = Path(__file__).parent.parent / "temp" / f"extracted_{job.id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Use the conversation extractor
        cmd = [
            "node", 
            str(extractor_path),
            actual_file_path,
            "--outdir", str(output_dir),
            "--format", "json"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Conversation extraction failed: {result.stderr}")
            return None
        
        logger.info(f"Conversation extraction completed: {result.stdout}")
        
        # Find the JSON output file
        json_files = list(output_dir.glob("*_extracted.json"))
        if not json_files:
            logger.error(f"No extracted JSON file found in {output_dir}")
            return None
        
        # Read the JSON output
        with open(json_files[0], "r") as f:
            extracted_data = json.load(f)
        
        # Clean up temporary files
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
        
        # Prepare combined content for chunking with section logging
        combined_content = ""
        sections_processed = 0
        
        # Add code blocks with language markers
        if extracted_data.get("code"):
            section_id = f"{job.id}_conv_code"
            code_content = extracted_data["code"]
            combined_content += "# Extracted Code\n\n" + code_content + "\n\n"
            sections_processed += 1
            
            if ENHANCED_MODULES_AVAILABLE:
                log_loop_record(section_id, [{"section": "code", "chars": len(code_content)}])
        
        # Add notes
        if extracted_data.get("notes"):
            section_id = f"{job.id}_conv_notes"
            notes_content = extracted_data["notes"]
            combined_content += "# Extracted Notes\n\n" + notes_content + "\n\n"
            sections_processed += 1
            
            if ENHANCED_MODULES_AVAILABLE:
                log_loop_record(section_id, [{"section": "notes", "chars": len(notes_content)}])
        
        # Add conversation
        if extracted_data.get("conversation"):
            section_id = f"{job.id}_conv_dialogue"
            conv_content = extracted_data["conversation"]
            combined_content += "# Conversation\n\n" + conv_content
            sections_processed += 1
            
            if ENHANCED_MODULES_AVAILABLE:
                log_loop_record(section_id, [{"section": "conversation", "chars": len(conv_content)}])
        
        logger.info(f"[Summary] Conversation extraction complete: {sections_processed} sections, {len(combined_content)} total characters")
        return combined_content
        
    except Exception as e:
        logger.exception(f"Error extracting conversation content: {str(e)}")
        return None

# Extract content from a text file
async def extract_text(file_path: Optional[str], file_content: Optional[bytes], job: IngestJob) -> Optional[str]:
    """
    Extract content from a text file with enhanced logging.
    
    Args:
        file_path: Path to the text file
        file_content: Raw text content if file_path is None
        job: The ingest job
        
    Returns:
        str: Extracted text content, or None if extraction failed
    """
    logger.info(f"[LoopRecord] Text extraction started: job_id={job.id}")
    
    try:
        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        elif file_content:
            content = file_content.decode("utf-8")
        else:
            logger.error("No file path or content provided")
            return None
        
        # Log extraction result
        segment_id = f"{job.id}_text_extraction"
        if ENHANCED_MODULES_AVAILABLE:
            log_loop_record(segment_id, [{"type": "text", "chars": len(content)}])
        
        logger.info(f"[Summary] Text extraction complete: {len(content)} characters")
        return content
        
    except Exception as e:
        logger.exception(f"Error extracting text content: {str(e)}")
        return None

# Chunk content into manageable pieces with enhanced logging
async def chunk_content(content: str, job: IngestJob) -> Optional[List[Chunk]]:
    """
    Chunk content into manageable pieces with enhanced segment tracking.
    
    Args:
        content: The text content to chunk
        job: The ingest job
        
    Returns:
        List[Chunk]: List of content chunks, or None if chunking failed
    """
    logger.info(f"[LoopRecord] Content chunking started: job_id={job.id}")
    
    try:
        # Determine media type for adaptive processing
        media_type = determine_media_type(getattr(job.request, 'source_url', None), None)
        
        # Split content into paragraphs
        paragraphs = content.split("\n\n")
        
        # Initialize chunk variables
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
            
            # If adding this paragraph would exceed chunk size, create a new chunk
            if len(current_chunk) + len(paragraph) > CHUNK_SIZE and current_chunk:
                # Create chunk
                chunk_id = f"{job.id}_{chunk_index}"
                chunk_text = current_chunk.strip()
                
                # Calculate chunk hash
                chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                
                chunks.append(Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    index=chunk_index,
                    sha256=chunk_hash,
                    start_offset=current_start,
                    end_offset=current_start + len(current_chunk),
                    metadata={
                        "job_id": job.id,
                        "document_type": job.request.document_type,
                        "source": job.request.source_url or "unknown",
                        "media_type": media_type,
                        "chunk_method": "paragraph_split"
                    }
                ))
                
                # Log chunk creation (Issue #1)
                segment_id = f"{job.id}_chunk_{chunk_index}"
                if ENHANCED_MODULES_AVAILABLE:
                    log_loop_record(segment_id, [{"chunk": chunk_index, "chars": len(chunk_text)}])
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-min(len(words), CHUNK_OVERLAP // 10):]
                current_chunk = " ".join(overlap_words) + "\n\n" + paragraph + "\n\n"
                current_start = current_start + len(current_chunk) - len(" ".join(overlap_words) + "\n\n")
                chunk_index += 1
            else:
                # Add paragraph to current chunk
                current_chunk += paragraph + "\n\n"
        
        # Add the last chunk if it's not empty
        if current_chunk.strip():
            chunk_id = f"{job.id}_{chunk_index}"
            chunk_text = current_chunk.strip()
            
            # Calculate chunk hash
            chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            
            chunks.append(Chunk(
                id=chunk_id,
                text=chunk_text,
                index=chunk_index,
                sha256=chunk_hash,
                start_offset=current_start,
                end_offset=current_start + len(current_chunk),
                metadata={
                    "job_id": job.id,
                    "document_type": job.request.document_type,
                    "source": job.request.source_url or "unknown",
                    "media_type": media_type,
                    "chunk_method": "paragraph_split"
                }
            ))
            
            # Log final chunk
            segment_id = f"{job.id}_chunk_{chunk_index}"
            if ENHANCED_MODULES_AVAILABLE:
                log_loop_record(segment_id, [{"chunk": chunk_index, "chars": len(chunk_text)}])
        
        logger.info(f"[Summary] Content chunking complete: {len(chunks)} chunks created")
        return chunks
        
    except Exception as e:
        logger.exception(f"Error chunking content: {str(e)}")
        return None

# Vectorize chunks with enhanced metadata
async def vectorize_chunks(chunks: List[Chunk], job: IngestJob) -> Optional[List[Tuple[Chunk, List[float]]]]:
    """
    Vectorize content chunks using a vector embedding model with enhanced tracking.
    
    Args:
        chunks: List of content chunks
        job: The ingest job
        
    Returns:
        List[Tuple[Chunk, List[float]]]: List of chunks and their vector embeddings,
            or None if vectorization failed
    """
    logger.info(f"[LoopRecord] Chunk vectorization started: job_id={job.id}, chunks={len(chunks)}")
    
    try:
        # In a real implementation, we would use a proper embedding model
        # This is a mock implementation that returns deterministic vectors
        import random
        vector_dim = config.get("scholar_sphere", {}).get("phase_vector_dim", 1024)
        
        results = []
        for i, chunk in enumerate(chunks):
            # Generate a deterministic random vector based on the chunk text
            random.seed(chunk.sha256)
            vector = [random.uniform(-1, 1) for _ in range(vector_dim)]
            
            # Normalize the vector
            norm = sum(x**2 for x in vector) ** 0.5
            vector = [x / norm for x in vector]
            
            results.append((chunk, vector))
            
            # Log vectorization progress
            segment_id = f"{job.id}_vector_{i}"
            if ENHANCED_MODULES_AVAILABLE:
                log_loop_record(segment_id, [{"chunk_id": chunk.id, "vector_dim": len(vector)}])
            
            # Update job progress
            job.chunks_processed += 1
            job.percent_complete = min(
                70.0,  # Cap at 70% since we still have concept mapping and storage
                10.0 + (job.chunks_processed / len(chunks) * 40.0)
            )
        
        logger.info(f"[Summary] Vectorization complete: {len(results)} chunks vectorized")
        return results
        
    except Exception as e:
        logger.exception(f"Error vectorizing chunks: {str(e)}")
        return None

# Map vectors to concepts with enhanced threshold management
async def map_to_concepts(vectors: List[Tuple[Chunk, List[float]]], job: IngestJob) -> Optional[List[ConceptVectorLink]]:
    """
    Map vector embeddings to concepts with enhanced threshold management and metadata.
    
    Args:
        vectors: List of chunks and their vector embeddings
        job: The ingest job
        
    Returns:
        List[ConceptVectorLink]: List of concept-vector links, or None if mapping failed
    """
    logger.info(f"[LoopRecord] Concept mapping started: job_id={job.id}, vectors={len(vectors)}")
    
    try:
        # Determine media type and content length for adaptive thresholds
        media_type = determine_media_type(getattr(job.request, 'source_url', None), None)
        total_content_length = sum(len(chunk.text) for chunk, _ in vectors)
        
        # Get adaptive threshold and max concepts
        if ENHANCED_MODULES_AVAILABLE:
            confidence_threshold = get_adaptive_threshold(total_content_length, media_type)
            max_concepts_per_chunk = get_fallback_count(10, media_type)  # Base candidate count
        else:
            confidence_threshold = get_threshold_for_media_type(media_type, MIN_CONFIDENCE)
            max_concepts_per_chunk = 8  # Fallback
        
        logger.info(f"[Config] Using threshold={confidence_threshold:.2f}, max_per_chunk={max_concepts_per_chunk} for {media_type}")
        
        import random
        links = []
        total_raw_concepts = 0
        total_filtered_concepts = 0
        
        for i, (chunk, vector) in enumerate(vectors):
            # Generate candidate concepts for this chunk
            num_candidates = random.randint(3, 12)  # Variable candidate count
            total_raw_concepts += num_candidates
            
            chunk_concepts = []
            for j in range(num_candidates):
                # Generate a deterministic concept ID and confidence
                concept_seed = f"{chunk.sha256}_{j}"
                random.seed(concept_seed)
                
                # Generate hex concept ID
                concept_id = ''.join(random.choices('0123456789abcdef', k=24))
                
                # Generate confidence with realistic distribution
                base_confidence = 0.3 + (random.random() * 0.7)  # 0.3 to 1.0
                
                # Apply media-type specific adjustments
                if media_type == "audio":
                    base_confidence *= 0.9  # Slightly lower for audio transcripts
                elif media_type == "conversation":
                    base_confidence *= 0.95  # Slightly lower for conversations
                
                chunk_concepts.append({
                    "concept_id": concept_id,
                    "confidence": base_confidence,
                    "method": f"{media_type}_extraction",
                    "source": {"chunk_index": i, "chunk_id": chunk.id}
                })
            
            # Apply confidence filtering with fallback (Issue #2)
            if ENHANCED_MODULES_AVAILABLE:
                filtered_concepts = filter_concepts(chunk_concepts, confidence_threshold, media_type, len(chunk.text))
                
                # Apply fallback if too few concepts
                if len(filtered_concepts) < FALLBACK_MIN_COUNT:
                    filtered_concepts = apply_confidence_fallback(filtered_concepts, chunk_concepts, FALLBACK_MIN_COUNT)
                    segment_id = f"{job.id}_concept_chunk_{i}"
                    logger.info(f"[{segment_id}] Applied confidence fallback: {len(chunk_concepts)} → {len(filtered_concepts)} concepts")
            else:
                # Simple fallback filtering
                filtered_concepts = [c for c in chunk_concepts if c["confidence"] >= confidence_threshold]
                if len(filtered_concepts) < FALLBACK_MIN_COUNT:
                    filtered_concepts = sorted(chunk_concepts, key=lambda x: x["confidence"], reverse=True)[:FALLBACK_MIN_COUNT]
            
            # Cap concepts per chunk (but with higher limit - Issue #3)
            if len(filtered_concepts) > max_concepts_per_chunk:
                filtered_concepts = sorted(filtered_concepts, key=lambda x: x["confidence"], reverse=True)[:max_concepts_per_chunk]
            
            total_filtered_concepts += len(filtered_concepts)
            
            # Create ConceptVectorLink objects with enhanced metadata
            for concept_data in filtered_concepts:
                links.append(ConceptVectorLink(
                    concept_id=concept_data["concept_id"],
                    chunk_id=chunk.id,
                    strength=concept_data["confidence"],
                    phase_vector=vector,
                    encoder_version=ENCODER_VERSION,
                    metadata={
                        "extraction_method": concept_data["method"],
                        "source_reference": concept_data["source"],
                        "media_type": media_type,
                        "confidence_threshold": confidence_threshold,
                        "chunk_text_length": len(chunk.text),
                        "extraction_timestamp": datetime.now().isoformat()
                    }
                ))
            
            # Log concept mapping for this chunk (Issue #1)
            segment_id = f"{job.id}_concept_chunk_{i}"
            if ENHANCED_MODULES_AVAILABLE:
                concept_summary = [{"concept_id": c["concept_id"][:8], "confidence": c["confidence"]} for c in filtered_concepts]
                log_loop_record(segment_id, concept_summary)
            
            # Update job progress
            job.percent_complete = min(
                90.0,  # Cap at 90% since we still have storage
                70.0 + (i / len(vectors) * 20.0)
            )
        
        # Validate concepts if enhanced modules available (Issue #4)
        if ENHANCED_MODULES_AVAILABLE:
            # Create validation-friendly format
            validation_concepts = []
            for link in links:
                validation_concepts.append({
                    "name": f"Concept_{link.concept_id[:8]}",
                    "confidence": link.strength,
                    "method": link.metadata.get("extraction_method", "unknown"),
                    "source": link.metadata.get("source_reference", {})
                })
            
            valid_count = validate_concepts(validation_concepts, f"{job.id}_final_concepts")
            logger.info(f"Concept validation: {valid_count}/{len(validation_concepts)} concepts valid")
        
        # Log final summary
        if ENHANCED_MODULES_AVAILABLE:
            log_concept_summary(job.id, [{"concept_id": link.concept_id, "strength": link.strength} for link in links])
        
        logger.info(f"[Summary] Concept mapping complete: {total_raw_concepts} candidates → {total_filtered_concepts} filtered → {len(links)} final concepts")
        return links
        
    except Exception as e:
        logger.exception(f"Error mapping to concepts: {str(e)}")
        return None

# Store results with enhanced metadata preservation
async def store_results(
    chunks: List[Chunk], 
    vectors: List[Tuple[Chunk, List[float]]], 
    concept_links: List[ConceptVectorLink],
    job: IngestJob
) -> bool:
    """
    Store the processed results with full metadata preservation.
    
    Args:
        chunks: List of content chunks
        vectors: List of chunks and their vector embeddings
        concept_links: List of concept-vector links
        job: The ingest job
        
    Returns:
        bool: True if storage succeeded, False otherwise
    """
    logger.info(f"[LoopRecord] Result storage started: job_id={job.id}")
    
    try:
        # Log storage attempt
        storage_summary = {
            "chunks": len(chunks),
            "vectors": len(vectors),
            "concept_links": len(concept_links),
            "job_id": job.id
        }
        
        logger.info(f"Storing results: {storage_summary}")
        
        # Create enhanced storage format
        enhanced_results = {
            "job_metadata": {
                "job_id": job.id,
                "document_type": job.request.document_type,
                "source_url": job.request.source_url,
                "processing_timestamp": datetime.now().isoformat(),
                "chunks_processed": len(chunks),
                "concepts_mapped": len(concept_links)
            },
            "chunks": [chunk.dict() for chunk in chunks],
            "concept_links": [
                {
                    **link.dict(),
                    "enhanced_metadata": True,
                    "storage_timestamp": datetime.now().isoformat()
                }
                for link in concept_links
            ],
            "processing_summary": storage_summary
        }
        
        # Save to output directory if specified
        output_dir = job.request.metadata.get("output_dir")
        if output_dir:
            output_path = Path(output_dir)
            os.makedirs(output_path, exist_ok=True)
            
            # Save enhanced results
            with open(output_path / f"{job.id}_enhanced_results.json", "w") as f:
                json.dump(enhanced_results, f, indent=2)
            
            # Save diagnostic-friendly format
            diagnostic_concepts = []
            for link in concept_links:
                diagnostic_concepts.append({
                    "name": f"Concept_{link.concept_id[:8]}",
                    "confidence": link.strength,
                    "method": link.metadata.get("extraction_method", "unknown"),
                    "source": link.metadata.get("source_reference", {}),
                    "concept_id": link.concept_id,
                    "extraction_timestamp": link.metadata.get("extraction_timestamp")
                })
            
            with open(output_path / f"{job.id}_semantic_concepts.json", "w") as f:
                json.dump(diagnostic_concepts, f, indent=2)
            
            logger.info(f"Results saved to {output_path}")
        
        # Inject into ConceptMesh if available (Issue #4 integration)
        if ENHANCED_MODULES_AVAILABLE and concept_links:
            try:
                concept_diff_data = {
                    "type": "ingest_bus_document",
                    "title": job.request.source_url or f"Job_{job.id}",
                    "concepts": [
                        {
                            "name": f"Concept_{link.concept_id[:8]}",
                            "confidence": link.strength,
                            "method": link.metadata.get("extraction_method", "unknown"),
                            "source": link.metadata.get("source_reference", {}),
                            "eigenfunction_id": f"ingest-{link.concept_id[:12]}"
                        }
                        for link in concept_links
                    ],
                    "summary": f"Ingest-bus processed {len(concept_links)} concepts",
                    "metadata": {
                        "job_id": job.id,
                        "document_type": str(job.request.document_type),
                        "processing_timestamp": datetime.now().isoformat(),
                        "chunk_count": len(chunks)
                    }
                }
                
                add_concept_diff(concept_diff_data)
                logger.info(f"Injected {len(concept_links)} concepts into ConceptMesh")
                
            except Exception as e:
                logger.warning(f"ConceptMesh injection failed: {e}")
        
        return True
        
    except Exception as e:
        logger.exception(f"Error storing results: {str(e)}")
        return False

# Send callback when job is complete (unchanged)
async def send_callback(job: IngestJob) -> bool:
    """
    Send a callback to the URL specified in the job request.
    
    Args:
        job: The completed ingest job
        
    Returns:
        bool: True if the callback succeeded, False otherwise
    """
    if not job.request.callback_url:
        return True
    
    logger.info(f"Sending callback to {job.request.callback_url}: job_id={job.id}")
    
    try:
        import requests
        
        response = requests.post(
            job.request.callback_url,
            json={
                "job_id": job.id,
                "status": job.status,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "chunks_processed": job.chunks_processed,
                "concepts_mapped": job.concepts_mapped,
                "chunk_ids": job.chunk_ids,
                "concept_ids": job.concept_ids,
                "enhanced_processing": ENHANCED_MODULES_AVAILABLE
            },
            timeout=10
        )
        
        if response.status_code >= 200 and response.status_code < 300:
            logger.info(f"Callback succeeded: {response.status_code}")
            return True
        else:
            logger.error(f"Callback failed: {response.status_code} {response.text}")
            return False
        
    except Exception as e:
        logger.exception(f"Error sending callback: {str(e)}")
        return False
