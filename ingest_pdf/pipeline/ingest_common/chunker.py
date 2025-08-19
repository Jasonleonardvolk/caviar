"""
ingest_common/chunker.py

Text chunking utilities for all media types.
Supports both fixed-size and adaptive chunking based on content.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    np = None

logger = logging.getLogger("chunker")

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    text: str
    index: int
    start_char: int
    end_char: int
    section: str = "body"
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "text": self.text,
            "index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "section": self.section,
            "metadata": self.metadata or {}
        }

# === Basic Chunking ===
def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separator: str = "\n\n"
) -> List[TextChunk]:
    """
    Split text into chunks of approximately equal size.
    
    Args:
        text: Input text to chunk
        chunk_size: Target size for each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        separator: Preferred separator for splitting
        
    Returns:
        List of TextChunk objects
    """
    if not text:
        return []
    
    chunks = []
    
    # Try to split on separator first
    if separator and separator in text:
        sections = text.split(separator)
        current_chunk = ""
        current_start = 0
        
        for section in sections:
            if len(current_chunk) + len(section) < chunk_size:
                current_chunk += section + separator
            else:
                if current_chunk:
                    chunks.append(TextChunk(
                        text=current_chunk.rstrip(),
                        index=len(chunks),
                        start_char=current_start,
                        end_char=current_start + len(current_chunk)
                    ))
                current_start += len(current_chunk)
                current_chunk = section + separator
        
        # Add final chunk
        if current_chunk:
            chunks.append(TextChunk(
                text=current_chunk.rstrip(),
                index=len(chunks),
                start_char=current_start,
                end_char=current_start + len(current_chunk)
            ))
    else:
        # Fall back to character-based chunking
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to find a sentence boundary
            if end < len(text):
                # Look for sentence endings
                for sep in [". ", "! ", "? ", "\n"]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start + chunk_size // 2:
                        end = last_sep + len(sep)
                        break
            
            chunk_text = text[start:end]
            chunks.append(TextChunk(
                text=chunk_text,
                index=len(chunks),
                start_char=start,
                end_char=end
            ))
            
            # Move start position with overlap
            start = end - chunk_overlap if end < len(text) else end
    
    return chunks

# === Sentence-Based Chunking ===
def chunk_by_sentences(
    text: str,
    sentences_per_chunk: int = 5,
    min_chunk_size: int = 100
) -> List[TextChunk]:
    """
    Split text into chunks based on sentence boundaries.
    
    Args:
        text: Input text
        sentences_per_chunk: Target number of sentences per chunk
        min_chunk_size: Minimum chunk size in characters
        
    Returns:
        List of TextChunk objects
    """
    # Simple sentence splitting (can be improved with NLTK/spaCy)
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    
    chunks = []
    current_sentences = []
    current_start = 0
    char_count = 0
    
    for sentence in sentences:
        current_sentences.append(sentence)
        char_count += len(sentence) + 1  # +1 for space
        
        # Check if we should create a chunk
        if (len(current_sentences) >= sentences_per_chunk and 
            char_count >= min_chunk_size):
            chunk_text = ' '.join(current_sentences)
            chunks.append(TextChunk(
                text=chunk_text,
                index=len(chunks),
                start_char=current_start,
                end_char=current_start + len(chunk_text)
            ))
            current_start += len(chunk_text) + 1
            current_sentences = []
            char_count = 0
    
    # Add remaining sentences
    if current_sentences:
        chunk_text = ' '.join(current_sentences)
        chunks.append(TextChunk(
            text=chunk_text,
            index=len(chunks),
            start_char=current_start,
            end_char=current_start + len(chunk_text)
        ))
    
    return chunks

# === Adaptive Entropy-Based Chunking ===
def calculate_text_entropy(text: str) -> float:
    """Calculate Shannon entropy of text"""
    if not text:
        return 0.0
    
    # Character frequency
    char_freq = {}
    for char in text.lower():
        if char.isalnum():
            char_freq[char] = char_freq.get(char, 0) + 1
    
    total_chars = sum(char_freq.values())
    if total_chars == 0:
        return 0.0
    
    # Calculate entropy
    entropy = 0.0
    for count in char_freq.values():
        prob = count / total_chars
        if prob > 0:
            entropy -= prob * np.log2(prob)
    
    return entropy

def adaptive_chunk_by_entropy(
    text: str,
    target_entropy: float = 4.0,
    min_chunk_size: int = 200,
    max_chunk_size: int = 2000
) -> List[TextChunk]:
    """
    Adaptively chunk text based on entropy changes.
    High entropy = more information = smaller chunks.
    Low entropy = repetitive = larger chunks.
    
    This mirrors the holographic interference pattern concept:
    - High entropy regions = high frequency waves = fine detail
    - Low entropy regions = low frequency waves = coarse structure
    
    Args:
        text: Input text
        target_entropy: Target entropy level
        min_chunk_size: Minimum chunk size
        max_chunk_size: Maximum chunk size
        
    Returns:
        List of TextChunk objects
    """
    chunks = []
    current_chunk = ""
    current_start = 0
    
    # Sliding window for entropy calculation
    window_size = min_chunk_size
    stride = window_size // 2
    
    i = 0
    while i < len(text):
        window_end = min(i + window_size, len(text))
        window_text = text[i:window_end]
        entropy = calculate_text_entropy(window_text)
        
        # Adjust chunk size based on entropy
        # High entropy = smaller chunks, Low entropy = larger chunks
        if entropy > 0:
            adaptive_size = int(target_entropy / entropy * min_chunk_size)
            adaptive_size = max(min_chunk_size, min(adaptive_size, max_chunk_size))
        else:
            adaptive_size = max_chunk_size
        
        # Extend current chunk
        chunk_end = min(i + adaptive_size, len(text))
        
        # Find good breaking point
        if chunk_end < len(text):
            for sep in ["\n\n", ". ", "\n", " "]:
                last_sep = text.rfind(sep, i, chunk_end)
                if last_sep > i + min_chunk_size:
                    chunk_end = last_sep + len(sep)
                    break
        
        chunk_text = text[i:chunk_end]
        
        chunks.append(TextChunk(
            text=chunk_text,
            index=len(chunks),
            start_char=i,
            end_char=chunk_end,
            metadata={"entropy": entropy}
        ))
        
        i = chunk_end
    
    return chunks

# === Semantic Chunking (Future) ===
def chunk_by_semantic_similarity(
    text: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    similarity_threshold: float = 0.8
) -> List[TextChunk]:
    """
    Chunk text based on semantic similarity between sentences.
    Groups similar content together.
    
    Note: Requires sentence-transformers library.
    """
    # TODO: Implement when adding transformer support
    logger.warning("Semantic chunking not yet implemented, falling back to sentence chunking")
    return chunk_by_sentences(text)

# === Media-Specific Chunking ===
def chunk_transcript(
    transcript: str,
    timestamps: Optional[List[Tuple[float, float]]] = None,
    target_duration: float = 30.0
) -> List[TextChunk]:
    """
    Chunk transcript with optional timestamp alignment.
    Used for audio/video transcripts.
    
    Args:
        transcript: Full transcript text
        timestamps: Optional list of (start_time, end_time) tuples
        target_duration: Target duration per chunk in seconds
        
    Returns:
        List of TextChunk objects with timing metadata
    """
    if timestamps:
        # Chunk based on timestamps
        chunks = []
        current_text = []
        current_start_time = 0
        current_start_char = 0
        
        for i, (start_time, end_time) in enumerate(timestamps):
            # Add text segment
            segment_text = transcript  # TODO: Extract segment
            current_text.append(segment_text)
            
            # Check if we should create a chunk
            duration = end_time - current_start_time
            if duration >= target_duration:
                chunk_text = ' '.join(current_text)
                chunks.append(TextChunk(
                    text=chunk_text,
                    index=len(chunks),
                    start_char=current_start_char,
                    end_char=current_start_char + len(chunk_text),
                    metadata={
                        "start_time": current_start_time,
                        "end_time": end_time,
                        "duration": duration
                    }
                ))
                current_text = []
                current_start_time = end_time
                current_start_char += len(chunk_text) + 1
        
        # Add remaining text
        if current_text:
            chunk_text = ' '.join(current_text)
            chunks.append(TextChunk(
                text=chunk_text,
                index=len(chunks),
                start_char=current_start_char,
                end_char=len(transcript),
                metadata={
                    "start_time": current_start_time,
                    "end_time": timestamps[-1][1] if timestamps else None
                }
            ))
        
        return chunks
    else:
        # Fall back to regular chunking
        return chunk_by_sentences(transcript)

# === Holographic-Inspired Chunking ===
def chunk_by_wave_interference(
    text: str,
    frequency: float = 1.0,
    amplitude: float = 1.0
) -> List[TextChunk]:
    """
    Wave interference chunking (requires NumPy).
    
    Experimental: Chunk text using wave interference patterns.
    Maps text position to wave phase for dynamic chunk boundaries.
    
    This creates a "holographic" chunking pattern where:
    - Constructive interference = chunk boundary
    - Destructive interference = keep together
    
    Args:
        text: Input text
        frequency: Wave frequency (higher = more chunks)
        amplitude: Wave amplitude (affects chunk size variation)
        
    Returns:
        List of TextChunk objects
    """
    if not _HAS_NUMPY:
        logger.warning("NumPy not available, falling back to sentence chunking")
        return chunk_by_sentences(text, sentences_per_chunk=5)
    chunks = []
    chunk_boundaries = [0]
    
    # Generate interference pattern
    for i in range(len(text)):
        # Primary wave
        wave1 = amplitude * np.sin(2 * np.pi * frequency * i / len(text))
        # Secondary wave (slightly different frequency for interference)
        wave2 = amplitude * np.sin(2 * np.pi * frequency * 1.1 * i / len(text))
        
        # Interference
        interference = wave1 + wave2
        
        # Detect constructive interference peaks
        if i > 0 and interference > amplitude * 1.5:
            # Check for suitable break point
            for sep in ["\n", ". ", " "]:
                if i < len(text) and text[i:i+len(sep)] == sep:
                    chunk_boundaries.append(i + len(sep))
                    break
    
    chunk_boundaries.append(len(text))
    
    # Create chunks from boundaries
    for i in range(len(chunk_boundaries) - 1):
        start = chunk_boundaries[i]
        end = chunk_boundaries[i + 1]
        
        if end > start:
            chunks.append(TextChunk(
                text=text[start:end],
                index=len(chunks),
                start_char=start,
                end_char=end,
                metadata={
                    "wave_phase": (2 * np.pi * frequency * start / len(text)) % (2 * np.pi)
                }
            ))
    
    return chunks

# === Utility Functions ===
def merge_small_chunks(chunks: List[TextChunk], min_size: int = 100) -> List[TextChunk]:
    """Merge chunks that are too small"""
    if not chunks:
        return chunks
    
    merged = []
    current = chunks[0]
    
    for next_chunk in chunks[1:]:
        if len(current.text) < min_size:
            # Merge with next
            current = TextChunk(
                text=current.text + " " + next_chunk.text,
                index=current.index,
                start_char=current.start_char,
                end_char=next_chunk.end_char,
                section=current.section,
                metadata={**current.metadata, **next_chunk.metadata} if current.metadata else next_chunk.metadata
            )
        else:
            merged.append(current)
            current = next_chunk
    
    merged.append(current)
    
    # Re-index
    for i, chunk in enumerate(merged):
        chunk.index = i
    
    return merged

logger.info("Chunker module loaded with holographic wave interference support")


