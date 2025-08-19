"""
ingest_common/utils.py

Common utilities shared across all ingestion handlers.
Extracted from the TORI pipeline for reuse.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable
from datetime import datetime

logger = logging.getLogger("ingest_common")

# === Safe Math Operations ===
def safe_num(value: Any, default: float = 0.0) -> float:
    """Safely convert any value to float"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with zero handling"""
    if denominator == 0:
        return default
    return numerator / denominator

def safe_multiply(a: float, b: float) -> float:
    """Safe multiplication"""
    return safe_num(a) * safe_num(b)

def safe_percentage(part: float, whole: float, default: float = 0.0) -> float:
    """Calculate percentage safely"""
    return safe_divide(part * 100, whole, default)

def safe_round(value: float, decimals: int = 2) -> float:
    """Safe rounding"""
    return round(safe_num(value), decimals)

def safe_get(data: Dict, key: str, default: Any = None) -> Any:
    """Safe dictionary get with None handling"""
    if data is None:
        return default
    return data.get(key, default)

def sanitize_dict(data: Dict) -> Dict:
    """Sanitize dictionary for JSON serialization"""
    if not isinstance(data, dict):
        return {}
    
    sanitized = {}
    for key, value in data.items():
        if value is None:
            continue
        elif isinstance(value, dict):
            sanitized[key] = sanitize_dict(value)
        elif isinstance(value, (list, tuple)):
            sanitized[key] = [sanitize_dict(item) if isinstance(item, dict) else item for item in value]
        else:
            sanitized[key] = value
    
    return sanitized

# === File Operations ===
def compute_sha256(file_path: Union[str, Path]) -> str:
    """Compute SHA-256 hash of file contents"""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.warning(f"Failed to compute SHA-256: {e}")
        return "error"

def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get basic file information"""
    path = Path(file_path)
    
    if not path.exists():
        return {
            "exists": False,
            "error": "File not found"
        }
    
    stat = path.stat()
    return {
        "exists": True,
        "filename": path.name,
        "file_path": str(path.absolute()),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "extension": path.suffix.lower(),
    }

# === Progress Tracking ===
class ProgressTracker:
    """Thread-safe progress tracking with deduplication"""
    
    def __init__(self, callback: Optional[Callable] = None):
        self.callback = callback
        self.last_stage = ""
        self.last_percent = -1
        
    async def send_progress(self, stage: str, percent: int, message: str):
        """Send progress update with deduplication"""
        # Skip duplicate updates
        if stage == self.last_stage and percent == self.last_percent:
            return
            
        self.last_stage = stage
        self.last_percent = percent
        
        if self.callback:
            try:
                if asyncio.iscoroutinefunction(self.callback):
                    await self.callback(stage, percent, message)
                else:
                    self.callback(stage, percent, message)
            except Exception as e:
                logger.debug(f"Progress callback error: {e}")

# Convenience function for backward compatibility
async def send_progress(callback: Optional[Callable], stage: str, percent: int, message: str):
    """Send progress update (legacy interface)"""
    if callback:
        tracker = ProgressTracker(callback)
        await tracker.send_progress(stage, percent, message)

# === DSP Utilities (from holographic engine) ===
def compute_spectral_features(audio_data: Any, sample_rate: int = 44100) -> Dict[str, Any]:
    """
    Compute spectral features from audio data.
    Placeholder for holographic DSP integration.
    """
    # TODO: Import from holographic engine
    return {
        "dominant_frequency": 440.0,
        "spectral_centroid": 1000.0,
        "spectral_rolloff": 8000.0,
        "spectral_flux": 0.1,
        "zero_crossing_rate": 0.05,
    }

def extract_motion_vectors(frame1: Any, frame2: Any) -> list:
    """
    Extract motion vectors between video frames.
    Placeholder for holographic engine integration.
    """
    # TODO: Import from holographic engine
    return [0.0, 0.0, 0.0]  # x, y, magnitude

# === Holographic Integration ===
def compute_psi_state(concepts: list, spectral_features: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Compute wavefunction state from concepts and features.
    Maps concept space to holographic representation.
    """
    # Basic psi state computation
    psi = {
        "amplitude": len(concepts) / 100.0,  # Normalize to 0-1
        "phase": 0.0,
        "frequency": 1.0,
        "coherence": 0.8,
    }
    
    # Incorporate spectral features if available
    if spectral_features:
        psi["frequency"] = spectral_features.get("dominant_frequency", 440.0) / 1000.0
        psi["phase"] = spectral_features.get("spectral_flux", 0.0) * 3.14159
    
    # Add concept-based modulation
    if concepts:
        avg_score = sum(c.get("score", 0.5) for c in concepts) / len(concepts)
        psi["coherence"] = avg_score
        
        # Section diversity affects phase spread
        sections = set(c.get("metadata", {}).get("section", "body") for c in concepts)
        psi["phase_spread"] = len(sections) / 10.0  # More sections = more phase spread
    
    return psi

# === Validation Utilities ===
def validate_file_size(file_path: Union[str, Path], max_size_mb: float) -> tuple:
    """Validate file size against limit"""
    info = get_file_info(file_path)
    
    if not info.get("exists"):
        return False, "File not found"
    
    size_mb = info.get("size_mb", 0)
    if size_mb > max_size_mb:
        return False, f"File too large: {size_mb:.1f}MB > {max_size_mb}MB limit"
    
    return True, "OK"

# === Import asyncio for async operations ===
import asyncio

# === Batch Processing Utilities ===
async def process_in_batches(items: list, batch_size: int, process_func: Callable) -> list:
    """Process items in batches to avoid memory issues"""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(*[process_func(item) for item in batch])
        results.extend(batch_results)
    
    return results

# === Time Utilities ===
def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

# === Resource Monitoring ===
import psutil
import os

def get_resource_usage() -> Dict[str, Any]:
    """Get current resource usage"""
    process = psutil.Process(os.getpid())
    
    return {
        "cpu_percent": process.cpu_percent(interval=0.1),
        "memory_mb": process.memory_info().rss / (1024 * 1024),
        "memory_percent": process.memory_percent(),
        "num_threads": process.num_threads(),
    }

logger.info("Common utilities loaded")


# ---------- Thread pool executor for run_sync --------------------------------
import atexit
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sync_exec")

def run_sync(func, *args, **kwargs):
    """Run blocking func in thread-pool when already inside an event-loop."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(_executor, func, *args, **kwargs)
    except RuntimeError:
        # No event loop, run directly
        return func(*args, **kwargs)

# Ensure clean shutdown for pytest
atexit.register(lambda: _executor.shutdown(wait=False))

# ---------- SHA-256 with chunk reading ---------------------------------------
def sha256_of_file(path: str, chunk_size: int = 8192) -> str:
    """Compute SHA-256 of file with chunked reading for large files."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

# Alias for backward compatibility
get_file_hash = sha256_of_file
