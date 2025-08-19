"""
optimized_io.py

Advanced I/O optimizations for PDF processing pipeline.
Implements streaming, caching, memory mapping, and async I/O.
"""

import os
import mmap
import hashlib
import time
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, BinaryIO
from dataclasses import dataclass, field
from functools import lru_cache
import zipfile
import gzip
import logging
from concurrent.futures import ThreadPoolExecutor
import json

# Get logger
logger = logging.getLogger("tori.ingest_pdf.optimized_io")

# Configuration
@dataclass
class IOConfig:
    """I/O optimization configuration."""
    chunk_size: int = 8192  # 8KB default, tunable
    enable_mmap: bool = True
    enable_digest_cache: bool = True
    cache_dir: Optional[str] = None
    sample_size: int = 8192
    max_async_files: int = 10
    enable_profiling: bool = False
    ssd_aligned: bool = True  # Align reads to 4KB boundaries
    
    def __post_init__(self):
        # Ensure chunk size is aligned for SSDs
        if self.ssd_aligned and self.chunk_size % 4096 != 0:
            self.chunk_size = ((self.chunk_size // 4096) + 1) * 4096
            logger.debug(f"Adjusted chunk_size to {self.chunk_size} for SSD alignment")

# Global configuration
default_config = IOConfig()

# I/O profiling
@dataclass
class IOProfile:
    """Track I/O performance metrics."""
    file_open_time: float = 0.0
    file_read_time: float = 0.0
    hash_compute_time: float = 0.0
    total_bytes_read: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_open_ms": round(self.file_open_time * 1000, 2),
            "file_read_ms": round(self.file_read_time * 1000, 2),
            "hash_compute_ms": round(self.hash_compute_time * 1000, 2),
            "total_mb_read": round(self.total_bytes_read / (1024 * 1024), 2),
            "throughput_mbps": round(
                (self.total_bytes_read / (1024 * 1024)) / max(0.001, self.file_read_time),
                2
            ),
            "cache_hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        }

# Thread-local profiling
import threading
_thread_local = threading.local()

def get_io_profile() -> IOProfile:
    """Get thread-local I/O profile."""
    if not hasattr(_thread_local, 'io_profile'):
        _thread_local.io_profile = IOProfile()
    return _thread_local.io_profile

# Streamed file hashing with profiling
def hash_file_stream(path: str, algo: str = 'sha256', 
                    config: Optional[IOConfig] = None) -> Tuple[str, int]:
    """
    Stream file hashing with optimizations.
    
    Returns:
        Tuple of (hash_hex, file_size)
    """
    if config is None:
        config = default_config
        
    profile = get_io_profile() if config.enable_profiling else None
    
    # Open file
    open_start = time.perf_counter()
    try:
        with open(path, 'rb') as f:
            if profile:
                profile.file_open_time += time.perf_counter() - open_start
                
            # Initialize hasher
            hash_start = time.perf_counter()
            hasher = hashlib.new(algo)
            file_size = 0
            
            # Read and hash in chunks
            read_start = time.perf_counter()
            while chunk := f.read(config.chunk_size):
                file_size += len(chunk)
                hasher.update(chunk)
                if profile:
                    profile.total_bytes_read += len(chunk)
                    
            if profile:
                profile.file_read_time += time.perf_counter() - read_start
                profile.hash_compute_time += time.perf_counter() - hash_start
                
            return hasher.hexdigest(), file_size
            
    except Exception as e:
        logger.error(f"Failed to hash file {path}: {e}")
        raise

# Memory-mapped file access
class MemoryMappedFile:
    """Context manager for memory-mapped file access."""
    
    def __init__(self, path: str, mode: str = 'r'):
        self.path = path
        self.mode = mode
        self.file = None
        self.mmap = None
        
    def __enter__(self) -> mmap.mmap:
        access_mode = 'rb' if 'r' in self.mode else 'r+b'
        self.file = open(self.path, access_mode)
        
        access = mmap.ACCESS_READ if 'r' in self.mode else mmap.ACCESS_WRITE
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=access)
        
        return self.mmap
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mmap:
            self.mmap.close()
        if self.file:
            self.file.close()

# Sample-based file analysis
def sample_file_content(path: str, config: Optional[IOConfig] = None) -> Dict[str, bytes]:
    """
    Sample file content for quick analysis.
    
    Returns dict with 'head', 'tail', and optionally 'middle' samples.
    """
    if config is None:
        config = default_config
        
    file_size = os.path.getsize(path)
    samples = {}
    
    with open(path, 'rb') as f:
        # Head sample
        samples['head'] = f.read(config.sample_size)
        
        # Tail sample if file is large enough
        if file_size > config.sample_size * 2:
            f.seek(-config.sample_size, 2)
            samples['tail'] = f.read(config.sample_size)
            
            # Middle sample for very large files
            if file_size > config.sample_size * 10:
                middle_pos = file_size // 2
                f.seek(middle_pos)
                samples['middle'] = f.read(config.sample_size)
                
    return samples

def estimate_file_entropy(samples: Dict[str, bytes]) -> float:
    """Estimate file entropy from samples."""
    # Combine all samples
    combined = b''.join(samples.values())
    
    # Calculate byte frequency
    byte_counts = {}
    total_bytes = len(combined)
    
    for byte in combined:
        byte_counts[byte] = byte_counts.get(byte, 0) + 1
        
    # Calculate Shannon entropy
    entropy = 0.0
    for count in byte_counts.values():
        if count > 0:
            probability = count / total_bytes
            entropy -= probability * (probability.bit_length() - 1)
            
    return entropy / 8.0  # Normalize to 0-1 range

# Content digest caching
class DigestCache:
    """Cache file digests to avoid repeated hashing."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or "./.digest_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}  # In-memory cache
        self._load_cache()
        
    def _load_cache(self):
        """Load cache from disk."""
        cache_file = self.cache_dir / "digest_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.memory_cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load digest cache: {e}")
                
    def _save_cache(self):
        """Save cache to disk."""
        cache_file = self.cache_dir / "digest_cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.memory_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save digest cache: {e}")
            
    def get_digest(self, path: str, algo: str = 'sha256') -> Optional[str]:
        """Get cached digest if available and file hasn't changed."""
        path = str(Path(path).resolve())
        stat = os.stat(path)
        
        cache_key = f"{path}:{algo}"
        cached = self.memory_cache.get(cache_key)
        
        if cached and cached['mtime'] == stat.st_mtime and cached['size'] == stat.st_size:
            profile = get_io_profile()
            if profile:
                profile.cache_hits += 1
            return cached['digest']
            
        profile = get_io_profile()
        if profile:
            profile.cache_misses += 1
        return None
        
    def set_digest(self, path: str, digest: str, algo: str = 'sha256'):
        """Cache a file digest."""
        path = str(Path(path).resolve())
        stat = os.stat(path)
        
        cache_key = f"{path}:{algo}"
        self.memory_cache[cache_key] = {
            'digest': digest,
            'mtime': stat.st_mtime,
            'size': stat.st_size,
            'cached_at': time.time()
        }
        
        # Periodically save to disk
        if len(self.memory_cache) % 100 == 0:
            self._save_cache()

# Global digest cache
_digest_cache = None

def get_digest_cache() -> DigestCache:
    """Get global digest cache instance."""
    global _digest_cache
    if _digest_cache is None:
        _digest_cache = DigestCache()
    return _digest_cache

# Cached hashing
def hash_file_cached(path: str, algo: str = 'sha256',
                    config: Optional[IOConfig] = None) -> str:
    """Hash file with caching."""
    if config is None:
        config = default_config
        
    if config.enable_digest_cache:
        cache = get_digest_cache()
        cached_digest = cache.get_digest(path, algo)
        if cached_digest:
            logger.debug(f"Cache hit for {path}")
            return cached_digest
            
    # Compute hash
    digest, _ = hash_file_stream(path, algo, config)
    
    # Cache result
    if config.enable_digest_cache:
        cache.set_digest(path, digest, algo)
        
    return digest

# Async file operations
async def read_file_async(path: str, chunk_size: Optional[int] = None) -> bytes:
    """Read file asynchronously."""
    if chunk_size is None:
        chunk_size = default_config.chunk_size
        
    chunks = []
    async with aiofiles.open(path, 'rb') as f:
        while chunk := await f.read(chunk_size):
            chunks.append(chunk)
            
    return b''.join(chunks)

async def hash_file_async(path: str, algo: str = 'sha256') -> Tuple[str, int]:
    """Hash file asynchronously."""
    hasher = hashlib.new(algo)
    file_size = 0
    
    async with aiofiles.open(path, 'rb') as f:
        while chunk := await f.read(default_config.chunk_size):
            file_size += len(chunk)
            hasher.update(chunk)
            
    return hasher.hexdigest(), file_size

async def process_files_async(paths: List[str], 
                            max_concurrent: Optional[int] = None) -> List[Dict[str, Any]]:
    """Process multiple files concurrently."""
    if max_concurrent is None:
        max_concurrent = default_config.max_async_files
        
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_one(path: str) -> Dict[str, Any]:
        async with semaphore:
            try:
                digest, size = await hash_file_async(path)
                return {
                    'path': path,
                    'digest': digest,
                    'size': size,
                    'status': 'success'
                }
            except Exception as e:
                return {
                    'path': path,
                    'status': 'error',
                    'error': str(e)
                }
                
    tasks = [process_one(path) for path in paths]
    return await asyncio.gather(*tasks)

# Compressed file size estimation
def estimate_zip_uncompressed_size(path: str) -> int:
    """Estimate uncompressed size of ZIP file without extraction."""
    try:
        with zipfile.ZipFile(path, 'r') as z:
            return sum(info.file_size for info in z.infolist())
    except Exception as e:
        logger.warning(f"Failed to estimate ZIP size: {e}")
        return 0

def estimate_gzip_uncompressed_size(path: str) -> int:
    """
    Estimate uncompressed size of GZIP file.
    Note: Only works for files < 4GB due to 32-bit size field.
    """
    try:
        with open(path, 'rb') as f:
            # Last 4 bytes contain uncompressed size (mod 2^32)
            f.seek(-4, 2)
            return int.from_bytes(f.read(4), 'little')
    except Exception as e:
        logger.warning(f"Failed to estimate GZIP size: {e}")
        return 0

# Optimized PDF metadata extraction
def extract_pdf_metadata_optimized(path: str, 
                                 config: Optional[IOConfig] = None) -> Dict[str, Any]:
    """Extract PDF metadata with I/O optimizations."""
    if config is None:
        config = default_config
        
    metadata = {
        'path': path,
        'filename': Path(path).name
    }
    
    # Get file hash (cached)
    metadata['sha256'] = hash_file_cached(path, config=config)
    
    # Get file size
    metadata['size'] = os.path.getsize(path)
    
    # Sample-based analysis
    samples = sample_file_content(path, config)
    metadata['entropy'] = estimate_file_entropy(samples)
    
    # Check if it's a valid PDF from header
    if samples['head'].startswith(b'%PDF'):
        metadata['valid_pdf'] = True
        # Extract version from header
        try:
            version_line = samples['head'].split(b'\n')[0]
            metadata['pdf_version'] = version_line.decode('ascii', errors='ignore')
        except:
            pass
    else:
        metadata['valid_pdf'] = False
        
    return metadata

# Benchmark utilities
class IOBenchmark:
    """Benchmark I/O operations."""
    
    def __init__(self):
        self.results = []
        
    def benchmark_read(self, path: str, chunk_sizes: List[int]) -> Dict[str, Any]:
        """Benchmark different chunk sizes."""
        file_size = os.path.getsize(path)
        results = {}
        
        for chunk_size in chunk_sizes:
            start = time.perf_counter()
            
            with open(path, 'rb') as f:
                bytes_read = 0
                while chunk := f.read(chunk_size):
                    bytes_read += len(chunk)
                    
            elapsed = time.perf_counter() - start
            throughput = (file_size / (1024 * 1024)) / elapsed  # MB/s
            
            results[f"{chunk_size}B"] = {
                'time_ms': round(elapsed * 1000, 2),
                'throughput_mbps': round(throughput, 2)
            }
            
        return results
        
    def find_optimal_chunk_size(self, path: str) -> int:
        """Find optimal chunk size for given file."""
        test_sizes = [4096, 8192, 16384, 32768, 65536, 131072]
        results = self.benchmark_read(path, test_sizes)
        
        # Find chunk size with best throughput
        best_size = max(results.items(), key=lambda x: x[1]['throughput_mbps'])
        logger.info(f"Optimal chunk size: {best_size[0]} ({best_size[1]['throughput_mbps']} MB/s)")
        
        return int(best_size[0].rstrip('B'))

# Factory function for optimized I/O
def create_io_config(
    chunk_size: Optional[int] = None,
    enable_cache: bool = True,
    enable_profiling: bool = False,
    cache_dir: Optional[str] = None
) -> IOConfig:
    """Create I/O configuration with custom settings."""
    return IOConfig(
        chunk_size=chunk_size or 8192,
        enable_digest_cache=enable_cache,
        enable_profiling=enable_profiling,
        cache_dir=cache_dir
    )

# Example usage wrapper
class OptimizedFileProcessor:
    """High-level file processor with all optimizations."""
    
    def __init__(self, config: Optional[IOConfig] = None):
        self.config = config or default_config
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def process_file(self, path: str) -> Dict[str, Any]:
        """Process single file with optimizations."""
        result = {
            'path': path,
            'metadata': extract_pdf_metadata_optimized(path, self.config)
        }
        
        # Add profiling data if enabled
        if self.config.enable_profiling:
            profile = get_io_profile()
            result['io_profile'] = profile.to_dict()
            
        return result
        
    async def process_files_batch(self, paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple files concurrently."""
        return await process_files_async(paths)
        
    def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)

# Set default configuration from environment
if os.environ.get('ENABLE_IO_PROFILING', '').lower() == 'true':
    default_config.enable_profiling = True
    
if os.environ.get('DISABLE_DIGEST_CACHE', '').lower() == 'true':
    default_config.enable_digest_cache = False
    
if chunk_size_env := os.environ.get('IO_CHUNK_SIZE'):
    default_config.chunk_size = int(chunk_size_env)
