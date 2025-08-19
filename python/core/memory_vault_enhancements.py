"""
Memory Vault Enhancement Patch - Compression and Multimodal Support
===================================================================

This patch adds compression and multimodal thumbnail support to the existing
memory vault implementation.
"""

import gzip
import zlib
import base64
from typing import Optional, Dict, Any

# Import thumbnail utilities
try:
    from .improved_memory_vault.thumbnail_utils import (
        ThumbnailGenerator,
        generate_image_thumbnail,
        generate_audio_fingerprint,
        generate_video_keyframes,
        generate_document_preview
    )
    THUMBNAIL_UTILS_AVAILABLE = True
except ImportError:
    THUMBNAIL_UTILS_AVAILABLE = False
    ThumbnailGenerator = None

# Configuration flags for compression
ENABLE_MEMORY_COMPRESSION = True
COMPRESSION_ALGORITHM = "gzip"  # Options: "gzip", "zlib", "bz2"
COMPRESSION_LEVEL = 6  # 1-9, where 9 is maximum compression
COMPRESS_THRESHOLD_BYTES = 1024  # Only compress if data > 1KB

class CompressionMixin:
    """
    Mixin class to add compression capabilities to memory vault.
    """
    
    def compress_data(self, data: bytes, algorithm: str = COMPRESSION_ALGORITHM) -> bytes:
        """
        Compress data using specified algorithm.
        
        Args:
            data: Raw bytes to compress
            algorithm: Compression algorithm to use
            
        Returns:
            Compressed bytes
        """
        if not ENABLE_MEMORY_COMPRESSION:
            return data
            
        if len(data) < COMPRESS_THRESHOLD_BYTES:
            return data  # Too small to benefit from compression
            
        try:
            if algorithm == "gzip":
                return gzip.compress(data, compresslevel=COMPRESSION_LEVEL)
            elif algorithm == "zlib":
                return zlib.compress(data, level=COMPRESSION_LEVEL)
            else:
                return data  # Unknown algorithm, return uncompressed
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return data
    
    def decompress_data(self, data: bytes, algorithm: str = COMPRESSION_ALGORITHM) -> bytes:
        """
        Decompress data using specified algorithm.
        
        Args:
            data: Compressed bytes
            algorithm: Compression algorithm used
            
        Returns:
            Decompressed bytes
        """
        if not ENABLE_MEMORY_COMPRESSION:
            return data
            
        try:
            if algorithm == "gzip":
                return gzip.decompress(data)
            elif algorithm == "zlib":
                return zlib.decompress(data)
            else:
                return data  # Unknown algorithm, assume uncompressed
        except Exception as e:
            # Data might not be compressed, return as is
            return data
    
    def compress_json(self, obj: Any) -> str:
        """
        Compress JSON object to base64 string.
        
        Args:
            obj: Object to serialize and compress
            
        Returns:
            Base64-encoded compressed JSON string
        """
        import json
        json_str = json.dumps(obj)
        json_bytes = json_str.encode('utf-8')
        compressed = self.compress_data(json_bytes)
        return base64.b64encode(compressed).decode('utf-8')
    
    def decompress_json(self, data: str) -> Any:
        """
        Decompress base64 JSON string to object.
        
        Args:
            data: Base64-encoded compressed JSON
            
        Returns:
            Decompressed object
        """
        import json
        compressed = base64.b64decode(data.encode('utf-8'))
        decompressed = self.decompress_data(compressed)
        json_str = decompressed.decode('utf-8')
        return json.loads(json_str)

class MultimodalMixin:
    """
    Mixin class to add multimodal thumbnail/fingerprint support to memory vault.
    """
    
    def __init__(self):
        """Initialize multimodal support."""
        self.thumbnail_generator = None
        if THUMBNAIL_UTILS_AVAILABLE:
            self.thumbnail_generator = ThumbnailGenerator()
    
    def process_multimodal_content(self, 
                                  content: Any,
                                  content_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Process multimodal content and generate appropriate thumbnails/fingerprints.
        
        Args:
            content: Content to process (path, bytes, or data)
            content_type: Type hint for content ('image', 'audio', 'video', 'document')
            
        Returns:
            Dictionary with processed metadata and thumbnails
        """
        if not THUMBNAIL_UTILS_AVAILABLE or not self.thumbnail_generator:
            return {'type': 'raw', 'processed': False}
        
        result = {'processed': True}
        
        try:
            # Auto-detect type from file extension if path provided
            if isinstance(content, (str, Path)) and Path(content).exists():
                file_path = Path(content)
                ext = file_path.suffix.lower()
                
                if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp'] or content_type == 'image':
                    result.update(self.thumbnail_generator.generate_image_thumbnail(image_path=str(file_path)))
                elif ext in ['.mp3', '.wav', '.flac', '.m4a'] or content_type == 'audio':
                    result.update(self.thumbnail_generator.generate_audio_fingerprint(audio_path=str(file_path)))
                elif ext in ['.mp4', '.avi', '.mov', '.mkv'] or content_type == 'video':
                    result.update(self.thumbnail_generator.generate_video_keyframes(str(file_path)))
                elif ext in ['.txt', '.md', '.py', '.js', '.html', '.pdf'] or content_type == 'document':
                    result.update(self.thumbnail_generator.generate_document_preview(document_path=str(file_path)))
                else:
                    result['type'] = 'unknown'
                    result['processed'] = False
            
            # Process raw data based on type hint
            elif content_type == 'image' and isinstance(content, bytes):
                result.update(self.thumbnail_generator.generate_image_thumbnail(image_data=content))
            elif content_type == 'audio' and isinstance(content, np.ndarray):
                result.update(self.thumbnail_generator.generate_audio_fingerprint(audio_data=content))
            elif content_type == 'document' and isinstance(content, str):
                result.update(self.thumbnail_generator.generate_document_preview(text_content=content))
            else:
                result['type'] = 'raw'
                result['processed'] = False
                
        except Exception as e:
            logger.error(f"Error processing multimodal content: {e}")
            result['processed'] = False
            result['error'] = str(e)
        
        return result
    
    def find_similar_images(self, perceptual_hash: str, threshold: int = 5) -> List[str]:
        """
        Find images with similar perceptual hashes.
        
        Args:
            perceptual_hash: Hash to search for
            threshold: Maximum hamming distance for similarity
            
        Returns:
            List of memory IDs with similar images
        """
        similar = []
        
        if not hasattr(self, 'memories'):
            return similar
        
        for memory_id, memory in self.memories.items():
            if hasattr(memory, 'multimodal_data') and memory.multimodal_data:
                if memory.multimodal_data.get('type') == 'image':
                    other_hash = memory.multimodal_data.get('perceptual_hash')
                    if other_hash and self._hamming_distance(perceptual_hash, other_hash) <= threshold:
                        similar.append(memory_id)
        
        return similar
    
    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        Calculate Hamming distance between two hashes.
        
        Args:
            hash1: First hash
            hash2: Second hash
            
        Returns:
            Hamming distance
        """
        if len(hash1) != len(hash2):
            return float('inf')
        
        # Convert hex strings to integers
        int1 = int(hash1, 16)
        int2 = int(hash2, 16)
        
        # XOR and count set bits
        xor = int1 ^ int2
        distance = 0
        while xor:
            distance += 1
            xor &= xor - 1  # Clear lowest set bit
        
        return distance

# Enhanced MemoryEntry with compression and multimodal support
def enhance_memory_entry(MemoryEntry):
    """
    Enhance existing MemoryEntry class with new capabilities.
    
    Args:
        MemoryEntry: Original MemoryEntry class
        
    Returns:
        Enhanced MemoryEntry class
    """
    
    class EnhancedMemoryEntry(MemoryEntry):
        """Enhanced memory entry with compression and multimodal support."""
        
        def __init__(self, *args, **kwargs):
            # Extract multimodal data if provided
            self.multimodal_data = kwargs.pop('multimodal_data', None)
            self.compressed = kwargs.pop('compressed', False)
            self.compression_algorithm = kwargs.pop('compression_algorithm', COMPRESSION_ALGORITHM)
            
            super().__init__(*args, **kwargs)
        
        def to_dict(self):
            """Convert to dictionary with enhanced fields."""
            data = super().to_dict() if hasattr(super(), 'to_dict') else asdict(self)
            data['multimodal_data'] = self.multimodal_data
            data['compressed'] = self.compressed
            data['compression_algorithm'] = self.compression_algorithm
            return data
        
        @classmethod
        def from_dict(cls, data):
            """Create from dictionary with enhanced fields."""
            multimodal_data = data.pop('multimodal_data', None)
            compressed = data.pop('compressed', False)
            compression_algorithm = data.pop('compression_algorithm', COMPRESSION_ALGORITHM)
            
            # Create base instance
            if hasattr(super(), 'from_dict'):
                instance = super().from_dict(data)
            else:
                instance = cls(**data)
            
            # Add enhanced fields
            instance.multimodal_data = multimodal_data
            instance.compressed = compressed
            instance.compression_algorithm = compression_algorithm
            
            return instance
    
    return EnhancedMemoryEntry

# Export enhancement functions
__all__ = [
    'CompressionMixin',
    'MultimodalMixin',
    'enhance_memory_entry',
    'ENABLE_MEMORY_COMPRESSION',
    'THUMBNAIL_UTILS_AVAILABLE'
]
