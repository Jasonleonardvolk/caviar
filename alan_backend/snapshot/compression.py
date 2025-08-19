# Copyright 2025 ALAN Team and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Patent Peace / Retaliation Notice:
#   As stated in Section 3 of the Apache 2.0 License, any entity that
#   initiates patent litigation (including a cross-claim or counterclaim)
#   alleging that this software or a contribution embodied within it
#   infringes a patent shall have all patent licenses granted herein
#   terminated as of the date such litigation is filed.

"""
Compression utilities for state snapshots.

This module provides compression and decompression functionality for snapshots,
using zstandard (zstd) for high-performance compression.
"""

import os
import sys
import logging
from enum import Enum, auto
from typing import Optional, Union, Tuple, Any

logger = logging.getLogger(__name__)

# Initialize zstd, with graceful fallback if not available
try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    logger.warning("zstandard module not available; compression disabled")
    HAS_ZSTD = False


# Magic identifier for compressed snapshots
COMPRESSED_MAGIC = b"ALSC"  # ALAN Snapshot Compressed

# Default compression level
DEFAULT_COMPRESSION_LEVEL = 3  # Good balance of speed and compression ratio

# Threshold for auto-compression (256 KB)
AUTO_COMPRESSION_THRESHOLD = 256 * 1024  # bytes


class CompressMode(Enum):
    """Compression mode for snapshots."""
    
    # Never compress
    NEVER = auto()
    
    # Always compress
    ALWAYS = auto()
    
    # Auto-decide based on size threshold
    AUTO = auto()


def should_compress(data: bytes, mode: CompressMode) -> bool:
    """Determine if data should be compressed based on mode and size.
    
    Args:
        data: Data to potentially compress
        mode: Compression mode
        
    Returns:
        True if data should be compressed, False otherwise
    """
    if not HAS_ZSTD:
        # Can't compress without zstd
        return False
        
    if mode == CompressMode.NEVER:
        return False
    elif mode == CompressMode.ALWAYS:
        return True
    else:  # mode == CompressMode.AUTO
        # Compress if data size exceeds threshold
        return len(data) >= AUTO_COMPRESSION_THRESHOLD


def compress(data: bytes, level: int = DEFAULT_COMPRESSION_LEVEL) -> bytes:
    """Compress data using zstd.
    
    Args:
        data: Data to compress
        level: Compression level (1-22, higher = better compression but slower)
        
    Returns:
        Compressed data with magic header
        
    Raises:
        ValueError: If zstd is not available
    """
    if not HAS_ZSTD:
        raise ValueError("zstandard module not available; compression not supported")
    
    # Create compressor with specified level
    compressor = zstd.ZstdCompressor(level=level)
    
    # Compress data
    compressed = compressor.compress(data)
    
    # Prepend magic identifier
    result = bytearray(COMPRESSED_MAGIC)
    result.extend(compressed)
    
    # Log compression stats
    original_size = len(data)
    compressed_size = len(result)
    ratio = original_size / compressed_size if compressed_size > 0 else 0
    logger.debug(
        f"Compressed {original_size} bytes to {compressed_size} bytes "
        f"(ratio: {ratio:.2f}x, level: {level})"
    )
    
    return bytes(result)


def decompress(data: bytes) -> bytes:
    """Decompress data compressed with zstd.
    
    Args:
        data: Compressed data with magic header
        
    Returns:
        Decompressed data
        
    Raises:
        ValueError: If zstd is not available or data is not properly compressed
    """
    if not HAS_ZSTD:
        raise ValueError("zstandard module not available; decompression not supported")
    
    # Check magic identifier
    if not data.startswith(COMPRESSED_MAGIC):
        raise ValueError(f"Data does not have compressed snapshot magic {COMPRESSED_MAGIC!r}")
    
    # Remove magic identifier
    compressed_data = data[len(COMPRESSED_MAGIC):]
    
    # Create decompressor
    decompressor = zstd.ZstdDecompressor()
    
    # Decompress data
    decompressed = decompressor.decompress(compressed_data)
    
    # Log decompression stats
    compressed_size = len(data)
    decompressed_size = len(decompressed)
    ratio = decompressed_size / compressed_size if compressed_size > 0 else 0
    logger.debug(
        f"Decompressed {compressed_size} bytes to {decompressed_size} bytes "
        f"(ratio: {ratio:.2f}x)"
    )
    
    return decompressed


def is_compressed(data: bytes) -> bool:
    """Check if data is compressed.
    
    Args:
        data: Data to check
        
    Returns:
        True if data starts with compression magic, False otherwise
    """
    return data.startswith(COMPRESSED_MAGIC)
