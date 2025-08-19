r"""
Penrose Tensor Acceleration Adapter
File: {PROJECT_ROOT}\python\core\penrose_adapter.py

High-performance Rust backend integration for fractal soliton memory operations.
Provides GPU-accelerated tensor operations for lattice evolution, phase entanglement,
and curvature field computations.

Architecture: PyO3/Rust backend with graceful Python fallback
"""

import numpy as np
import logging
import asyncio
import time
import threading
import concurrent.futures
from typing import Optional, Dict, Any, Tuple, Union, List, Literal, Callable
from dataclasses import dataclass, field
import json
import os
import ctypes.util
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import functools
import uuid

logger = logging.getLogger(__name__)

# Thread safety for singleton pattern
_singleton_lock = threading.Lock()

# Runtime detection flags
PENROSE_ENGINE_AVAILABLE = False
PENROSE_LIB = None

# Track consecutive fallbacks for logging
_consecutive_fallbacks = 0
_FALLBACK_WARNING_THRESHOLD = 5  # Log warning after this many consecutive fallbacks

@dataclass
class PenroseConfig:
    """Configuration for Penrose acceleration backend"""
    enable_gpu: bool = True
    max_threads: int = 8
    cache_size_mb: int = 512
    precision: Literal["float32", "float64"] = "float32"
    fallback_to_numpy: bool = True
    log_performance: bool = True
    
    def __eq__(self, other):
        """Compare configs for equality to detect config shadowing"""
        if not isinstance(other, PenroseConfig):
            return False
        return (self.enable_gpu == other.enable_gpu and
                self.max_threads == other.max_threads and
                self.cache_size_mb == other.cache_size_mb and
                self.precision == other.precision and
                self.fallback_to_numpy == other.fallback_to_numpy and
                self.log_performance == other.log_performance)

# Performance timing decorator
def _timer(func):
    """Decorator to measure execution time of a function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms
    return wrapper

class PenroseAdapter:
    """
    Singleton adapter for Rust-accelerated tensor operations.
    
    Provides high-performance implementations of:
    - Lattice field evolution (JIT-compiled)
    - Phase entanglement calculations
    - Curvature-to-phase encoding
    - Gradient computations
    - Memory compression (SVD/PCA)
    """
    
    _instance: Optional['PenroseAdapter'] = None
    _initialized: bool = False
    
    def __init__(self, config: Optional[PenroseConfig] = None):
        """
        Initialize the Penrose adapter with configuration
        
        Note: This should not be called directly. Use get_instance() instead.
        """
        if PenroseAdapter._initialized:
            return
            
        self.config = config or PenroseConfig()
        self.rust_lib = None
        self.performance_stats = {
            "operations_count": 0,
            "total_speedup": 0.0,
            "cache_hits": 0,
            "fallback_count": 0,
            "avg_rust_ms": 0.0,
            "avg_numpy_ms": 0.0,
            "last_update": time.time()
        }
        
        # Create dedicated thread pool executor
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_threads,
            thread_name_prefix="penrose_worker"
        )
        
        # Attempt to load Rust backend
        self._initialize_rust_backend()
        PenroseAdapter._initialized = True
    
    @classmethod
    def get_instance(cls, config: Optional[PenroseConfig] = None) -> 'PenroseAdapter':
        """
        Get singleton instance of Penrose adapter in a thread-safe manner
        
        Args:
            config: Optional configuration parameters
            
        Returns:
            The singleton instance of PenroseAdapter
        """
        with _singleton_lock:
            if cls._instance is None:
                cls._instance = cls(config)
            elif config is not None and config != cls._instance.config:
                logger.warning("PenroseAdapter already initialized; new config ignored")
                
        return cls._instance
    
    def _initialize_rust_backend(self) -> None:
        """Initialize the Rust backend library with robust error handling"""
        global PENROSE_ENGINE_AVAILABLE, PENROSE_LIB
        
        try:
            # Look for the library using ctypes.util.find_library
            lib_path = ctypes.util.find_library("penrose_engine")
            if lib_path:
                logger.debug(f"Found Penrose engine library at: {lib_path}")
            
            # Attempt to import the compiled Rust library
            import penrose_engine
            PENROSE_LIB = penrose_engine
            PENROSE_ENGINE_AVAILABLE = True
            
            # Initialize Rust engine with config
            init_result = PENROSE_LIB.initialize_engine(
                max_threads=self.config.max_threads,
                cache_size_mb=self.config.cache_size_mb,
                enable_gpu=self.config.enable_gpu,
                precision=self.config.precision
            )
            
            if init_result.get("success", False):
                logger.info(f"✅ Penrose tensor acceleration enabled")
                logger.info(f"   GPU: {init_result.get('gpu_available', False)}")
                logger.info(f"   Threads: {init_result.get('thread_count', 'unknown')}")
                logger.info(f"   Cache: {self.config.cache_size_mb}MB")
            else:
                raise RuntimeError(f"Rust engine initialization failed: {init_result.get('error', 'unknown')}")
                
        except ImportError as e:
            logger.warning("⚠️ Penrose Rust engine not available - using NumPy fallback")
            logger.debug(f"Import error: {e}")
            
            # Provide a helpful error message about PYTHONPATH
            env_path = os.environ.get('PYTHONPATH', '')
            logger.debug(f"PYTHONPATH: {env_path}")
            logger.debug("Make sure the Penrose engine library is in your PYTHONPATH or installed correctly")
            
            PENROSE_ENGINE_AVAILABLE = False
            
        except Exception as e:
            logger.error(f"❌ Penrose engine initialization error: {e}")
            PENROSE_ENGINE_AVAILABLE = False
            
            if not self.config.fallback_to_numpy:
                raise RuntimeError(f"Penrose required but unavailable: {e}")
    
    def is_available(self) -> bool:
        """Check if Penrose acceleration is available"""
        return PENROSE_ENGINE_AVAILABLE
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get detailed backend information"""
        if not PENROSE_ENGINE_AVAILABLE:
            return {
                "available": False,
                "backend": "numpy_fallback",
                "reason": "rust_engine_not_loaded"
            }
        
        try:
            info = PENROSE_LIB.get_engine_info()
            return {
                "available": True,
                "backend": "rust_penrose",
                "version": info.get("version", "unknown"),
                "gpu_enabled": info.get("gpu_enabled", False),
                "thread_count": info.get("thread_count", 1),
                "cache_size": info.get("cache_size_mb", 0),
                "precision": self.config.precision
            }
        except Exception as e:
            logger.warning(f"Could not get backend info: {e}")
            return {"available": False, "error": str(e)}
    
    # Helper method to estimate NumPy execution time based on input size
    def _estimate_numpy_time(self, shape_info: Tuple) -> float:
        """
        Estimate NumPy execution time for performance comparison
        
        Args:
            shape_info: Tuple describing the size of the input data
            
        Returns:
            Estimated execution time in milliseconds
        """
        # This is a very simple heuristic - can be made more sophisticated
        total_size = np.prod(shape_info)
        # Base time plus size-dependent component (ms)
        return 0.5 + total_size * 0.0001
    
    # Helper to manage fallback logging
    def _log_fallback(self, operation: str, error: Exception, req_id: Optional[str] = None):
        """
        Log fallback events with appropriate level based on frequency
        
        Args:
            operation: Name of the operation that failed
            error: Exception that caused the fallback
            req_id: Optional request ID for tracing
        """
        global _consecutive_fallbacks
        
        _consecutive_fallbacks += 1
        self.performance_stats["fallback_count"] += 1
        
        id_str = f" [req:{req_id}]" if req_id else ""
        
        if _consecutive_fallbacks >= _FALLBACK_WARNING_THRESHOLD:
            logger.warning(f"Rust {operation} failed {_consecutive_fallbacks} times in a row{id_str}: {error}")
            _consecutive_fallbacks = 0  # Reset after logging a warning
        else:
            logger.debug(f"Rust {operation} failed, using NumPy fallback{id_str}: {error}")
    
    async def evolve_lattice_field(
        self, 
        lattice: np.ndarray, 
        phase_field: np.ndarray,
        curvature_field: np.ndarray,
        dt: float = 0.01,
        req_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Accelerated lattice field evolution using Rust backend.
        
        Args:
            lattice: Current lattice state (H x W)
            phase_field: ψ-phase field (H x W) 
            curvature_field: Geometric curvature (H x W)
            dt: Time step for evolution
            req_id: Optional request ID for tracing
            
        Returns:
            Updated lattice field
        """
        global _consecutive_fallbacks
        
        if PENROSE_ENGINE_AVAILABLE:
            try:
                # Call Rust implementation with timing
                @_timer
                def rust_evolve():
                    result = PENROSE_LIB.evolve_lattice_field(
                        lattice.astype(self.config.precision),
                        phase_field.astype(self.config.precision),
                        curvature_field.astype(self.config.precision),
                        dt
                    )
                    # Check if result came from cache
                    if hasattr(result, 'from_cache') and result.from_cache:
                        self.performance_stats["cache_hits"] += 1
                    return result
                
                # Execute in our dedicated thread pool
                result, rust_ms = await asyncio.get_running_loop().run_in_executor(
                    self._executor, rust_evolve
                )
                
                # Update performance stats
                self.performance_stats["operations_count"] += 1
                self.performance_stats["avg_rust_ms"] = (
                    (self.performance_stats["avg_rust_ms"] * (self.performance_stats["operations_count"] - 1) + rust_ms) / 
                    self.performance_stats["operations_count"]
                )
                
                # Estimate NumPy time for comparison
                numpy_ms = self._estimate_numpy_time(lattice.shape)
                self.performance_stats["avg_numpy_ms"] = (
                    (self.performance_stats["avg_numpy_ms"] * (self.performance_stats["operations_count"] - 1) + numpy_ms) / 
                    self.performance_stats["operations_count"]
                )
                
                # Update speedup
                if rust_ms > 0:
                    self.performance_stats["total_speedup"] += numpy_ms / rust_ms
                
                # Reset consecutive fallbacks counter on success
                _consecutive_fallbacks = 0
                
                return np.array(result, dtype=lattice.dtype)
                
            except Exception as e:
                self._log_fallback("lattice evolution", e, req_id)
        
        # NumPy fallback implementation
        @_timer
        def numpy_fallback():
            return self._evolve_lattice_numpy(lattice, phase_field, curvature_field, dt)
        
        result, numpy_ms = numpy_fallback()
        
        # Update NumPy timing stats
        self.performance_stats["operations_count"] += 1
        self.performance_stats["avg_numpy_ms"] = (
            (self.performance_stats["avg_numpy_ms"] * (self.performance_stats["operations_count"] - 1) + numpy_ms) / 
            self.performance_stats["operations_count"]
        )
        
        return result
    
    def _evolve_lattice_numpy(
        self, 
        lattice: np.ndarray, 
        phase_field: np.ndarray,
        curvature_field: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Optimized NumPy fallback for lattice evolution
        
        Args:
            lattice: Current lattice state
            phase_field: Phase field
            curvature_field: Curvature field
            dt: Time step
            
        Returns:
            Evolved lattice field
        """
        # Compute gradients more efficiently
        h, w = phase_field.shape
        
        # Compute horizontal and vertical gradients with np.diff
        dx = np.zeros_like(phase_field)
        dy = np.zeros_like(phase_field)
        
        # Horizontal gradient (faster than np.gradient)
        dx[:, 1:] = phase_field[:, 1:] - phase_field[:, :-1]
        dx[:, 0] = phase_field[:, 0] - phase_field[:, -1]  # Wrap around
        
        # Vertical gradient
        dy[1:, :] = phase_field[1:, :] - phase_field[:-1, :]
        dy[0, :] = phase_field[0, :] - phase_field[-1, :]  # Wrap around
        
        # Square gradients (avoid allocation with inplace operation)
        dx_squared = dx**2
        dy_squared = dy**2
        
        # Compute evolution term
        curvature_coupling = curvature_field * np.cos(phase_field)
        evolution_term = 0.1 * (dx_squared + dy_squared) + 0.05 * curvature_coupling
        
        # Update lattice
        return lattice + dt * evolution_term
    
    async def compute_phase_entanglement(
        self,
        soliton_positions: np.ndarray,
        phases: np.ndarray,
        coupling_strength: float = 1.0,
        req_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute quantum phase entanglement between solitons.
        
        Args:
            soliton_positions: Soliton positions (N x 2)
            phases: Soliton phases (N,)
            coupling_strength: Entanglement coupling parameter
            req_id: Optional request ID for tracing
            
        Returns:
            Entanglement matrix (N x N)
        """
        global _consecutive_fallbacks
        
        if PENROSE_ENGINE_AVAILABLE:
            try:
                # Call Rust implementation with timing
                @_timer
                def rust_entangle():
                    result = PENROSE_LIB.compute_phase_entanglement(
                        soliton_positions.astype(self.config.precision),
                        phases.astype(self.config.precision),
                        coupling_strength
                    )
                    # Check if result came from cache
                    if hasattr(result, 'from_cache') and result.from_cache:
                        self.performance_stats["cache_hits"] += 1
                    return result
                
                # Execute in our dedicated thread pool
                result, rust_ms = await asyncio.get_running_loop().run_in_executor(
                    self._executor, rust_entangle
                )
                
                # Update performance stats
                self.performance_stats["operations_count"] += 1
                self.performance_stats["avg_rust_ms"] = (
                    (self.performance_stats["avg_rust_ms"] * (self.performance_stats["operations_count"] - 1) + rust_ms) / 
                    self.performance_stats["operations_count"]
                )
                
                # Estimate NumPy time for comparison (quadratic in N for entanglement)
                n = len(soliton_positions)
                numpy_ms = self._estimate_numpy_time((n, n)) * 10  # Entanglement is more expensive
                self.performance_stats["avg_numpy_ms"] = (
                    (self.performance_stats["avg_numpy_ms"] * (self.performance_stats["operations_count"] - 1) + numpy_ms) / 
                    self.performance_stats["operations_count"]
                )
                
                # Update speedup
                if rust_ms > 0:
                    self.performance_stats["total_speedup"] += numpy_ms / rust_ms
                
                # Reset consecutive fallbacks counter on success
                _consecutive_fallbacks = 0
                
                return np.array(result, dtype=phases.dtype)
                
            except Exception as e:
                self._log_fallback("entanglement computation", e, req_id)
        
        # NumPy fallback implementation with timing
        @_timer
        def numpy_fallback():
            return self._compute_entanglement_numpy(soliton_positions, phases, coupling_strength)
        
        result, numpy_ms = numpy_fallback()
        
        # Update NumPy timing stats
        self.performance_stats["operations_count"] += 1
        self.performance_stats["avg_numpy_ms"] = (
            (self.performance_stats["avg_numpy_ms"] * (self.performance_stats["operations_count"] - 1) + numpy_ms) / 
            self.performance_stats["operations_count"]
        )
        
        return result
    
    def _compute_entanglement_numpy(
        self,
        positions: np.ndarray,
        phases: np.ndarray,
        coupling: float
    ) -> np.ndarray:
        """
        Vectorized NumPy implementation for phase entanglement
        
        Args:
            positions: Soliton positions (N x 2)
            phases: Soliton phases (N,)
            coupling: Coupling strength
            
        Returns:
            Entanglement matrix (N x N)
        """
        # Vectorized implementation using broadcasting
        # Calculate pairwise distances between all positions
        diff = positions[:, None, :] - positions[None, :, :]  # Shape: (N, N, 2)
        distances = np.linalg.norm(diff, axis=2) + 1e-9  # Shape: (N, N), avoid division by zero
        
        # Calculate phase differences
        phase_diff = np.abs(phases[:, None] - phases[None, :])  # Shape: (N, N)
        
        # Calculate entanglement
        entanglement = coupling * np.exp(-distances / 10.0) * np.cos(phase_diff)
        
        # Zero out diagonal (no self-entanglement)
        np.fill_diagonal(entanglement, 0.0)
        
        return entanglement
    
    async def accelerated_curvature_encode(
        self,
        curvature_field: np.ndarray,
        encoding_mode: str = "log_phase",
        req_id: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Accelerated curvature-to-phase encoding.
        
        Args:
            curvature_field: Kretschmann scalar field (H x W)
            encoding_mode: "log_phase", "tanh", or "direct"
            req_id: Optional request ID for tracing
            
        Returns:
            Tuple of (phase_field, amplitude_field)
        """
        global _consecutive_fallbacks
        
        if PENROSE_ENGINE_AVAILABLE:
            try:
                # Call Rust implementation with timing
                @_timer
                def rust_encode():
                    result = PENROSE_LIB.curvature_to_phase_encode(
                        curvature_field.astype(self.config.precision),
                        encoding_mode
                    )
                    # Check if result came from cache
                    if hasattr(result, 'from_cache') and result.from_cache:
                        self.performance_stats["cache_hits"] += 1
                    return result
                
                # Execute in our dedicated thread pool
                result, rust_ms = await asyncio.get_running_loop().run_in_executor(
                    self._executor, rust_encode
                )
                
                # Update performance stats
                self.performance_stats["operations_count"] += 1
                self.performance_stats["avg_rust_ms"] = (
                    (self.performance_stats["avg_rust_ms"] * (self.performance_stats["operations_count"] - 1) + rust_ms) / 
                    self.performance_stats["operations_count"]
                )
                
                # Estimate NumPy time for comparison
                numpy_ms = self._estimate_numpy_time(curvature_field.shape)
                self.performance_stats["avg_numpy_ms"] = (
                    (self.performance_stats["avg_numpy_ms"] * (self.performance_stats["operations_count"] - 1) + numpy_ms) / 
                    self.performance_stats["operations_count"]
                )
                
                # Update speedup
                if rust_ms > 0:
                    self.performance_stats["total_speedup"] += numpy_ms / rust_ms
                
                # Reset consecutive fallbacks counter on success
                _consecutive_fallbacks = 0
                
                phase_field = np.array(result["phase"], dtype=curvature_field.dtype)
                amplitude_field = np.array(result["amplitude"], dtype=curvature_field.dtype)
                
                return phase_field, amplitude_field
                
            except Exception as e:
                self._log_fallback("curvature encoding", e, req_id)
        
        # NumPy fallback implementation with timing
        @_timer
        def numpy_fallback():
            return self._encode_curvature_numpy(curvature_field, encoding_mode)
        
        result, numpy_ms = numpy_fallback()
        
        # Update NumPy timing stats
        self.performance_stats["operations_count"] += 1
        self.performance_stats["avg_numpy_ms"] = (
            (self.performance_stats["avg_numpy_ms"] * (self.performance_stats["operations_count"] - 1) + numpy_ms) / 
            self.performance_stats["operations_count"]
        )
        
        return result
    
    def _encode_curvature_numpy(
        self,
        curvature: np.ndarray,
        mode: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimized NumPy fallback for curvature encoding
        
        Args:
            curvature: Curvature field
            mode: Encoding mode
            
        Returns:
            Tuple of (phase_field, amplitude_field)
        """
        if mode == "log_phase":
            # Avoid temporary arrays by using inplace operations where possible
            safe_curvature = np.maximum(curvature, 1e-8)
            phase = np.log(safe_curvature) % (2 * np.pi) - np.pi
            amplitude = 1.0 / (1.0 + safe_curvature**2)
        elif mode == "tanh":
            phase = np.tanh(curvature) * np.pi
            amplitude = 1.0 / np.cosh(curvature)
        else:  # direct
            phase = curvature % (2 * np.pi) - np.pi
            amplitude = np.ones_like(curvature)
        
        return phase, amplitude
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics
        
        Returns:
            Dictionary of performance metrics
        """
        stats = self.performance_stats.copy()
        
        if stats["operations_count"] > 0:
            stats["average_speedup"] = stats["total_speedup"] / stats["operations_count"]
            stats["fallback_rate"] = stats["fallback_count"] / stats["operations_count"]
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["operations_count"]
        else:
            stats["average_speedup"] = 0.0
            stats["fallback_rate"] = 0.0
            stats["cache_hit_rate"] = 0.0
            
        stats["backend_available"] = PENROSE_ENGINE_AVAILABLE
        stats["uptime_seconds"] = time.time() - stats["last_update"]
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset performance statistics"""
        self.performance_stats = {
            "operations_count": 0,
            "total_speedup": 0.0,
            "cache_hits": 0,
            "fallback_count": 0,
            "avg_rust_ms": 0.0,
            "avg_numpy_ms": 0.0,
            "last_update": time.time()
        }
    
    async def update_config(self, new_config: PenroseConfig) -> bool:
        """
        Update configuration and reinitialize the engine if needed
        
        Args:
            new_config: New configuration parameters
            
        Returns:
            True if successful, False otherwise
        """
        if new_config == self.config:
            logger.debug("Config unchanged, skipping update")
            return True
        
        logger.info("Updating Penrose configuration")
        
        # Shutdown existing engine
        await self.shutdown()
        
        # Update config
        self.config = new_config
        
        # Reinitialize
        with _singleton_lock:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.max_threads,
                thread_name_prefix="penrose_worker"
            )
            self._initialize_rust_backend()
        
        return PENROSE_ENGINE_AVAILABLE or self.config.fallback_to_numpy
    
    async def shutdown(self) -> None:
        """
        Clean shutdown of Penrose engine
        
        Ensures proper cleanup of resources, including:
        - Rust engine shutdown
        - Thread pool shutdown
        - Global state reset
        """
        global PENROSE_ENGINE_AVAILABLE, PENROSE_LIB
        
        logger.info("Shutting down Penrose adapter...")
        
        # Shutdown Rust engine if available
        if PENROSE_ENGINE_AVAILABLE and PENROSE_LIB:
            try:
                await asyncio.get_running_loop().run_in_executor(
                    self._executor,
                    PENROSE_LIB.shutdown_engine
                )
                logger.info("✅ Penrose engine shutdown complete")
            except Exception as e:
                logger.warning(f"Penrose shutdown warning: {e}")
        
        # Shutdown thread pool
        try:
            self._executor.shutdown(wait=True)
            logger.debug("Thread pool executor shutdown complete")
        except Exception as e:
            logger.warning(f"Thread pool shutdown error: {e}")
        
        # Reset global state
        PENROSE_ENGINE_AVAILABLE = False
        PENROSE_LIB = None
        
        # Reset singleton instance
        with _singleton_lock:
            PenroseAdapter._instance = None
            PenroseAdapter._initialized = False


# Convenience functions for direct usage
async def get_penrose_adapter(config: Optional[PenroseConfig] = None) -> PenroseAdapter:
    """
    Get the global Penrose adapter instance
    
    Args:
        config: Optional configuration
        
    Returns:
        The singleton PenroseAdapter instance
    """
    return PenroseAdapter.get_instance(config)

def is_penrose_available() -> bool:
    """
    Quick check if Penrose acceleration is available
    
    Returns:
        True if Penrose engine is available, False otherwise
    """
    return PENROSE_ENGINE_AVAILABLE

def get_penrose_info() -> Dict[str, Any]:
    """
    Get Penrose backend information
    
    Returns:
        Dictionary with backend information
    """
    adapter = PenroseAdapter.get_instance()
    return adapter.get_backend_info()

async def generate_request_id() -> str:
    """
    Generate a unique request ID for tracing
    
    Returns:
        Unique request ID string
    """
    return str(uuid.uuid4())

# Export main classes and functions
__all__ = [
    'PenroseAdapter',
    'PenroseConfig', 
    'get_penrose_adapter',
    'is_penrose_available',
    'get_penrose_info',
    'generate_request_id'
]

# Register FastAPI integration if available
try:
    from fastapi import FastAPI
    
    def register_penrose_shutdown_hook(app: FastAPI):
        """Register shutdown hook with FastAPI"""
        @app.on_event("shutdown")
        async def shutdown_hook():
            logger.info("FastAPI shutting down, cleaning up Penrose adapter")
            adapter = PenroseAdapter.get_instance()
            await adapter.shutdown()
        
        logger.info("Registered Penrose adapter shutdown hook with FastAPI")
    
    __all__.append('register_penrose_shutdown_hook')
except ImportError:
    # FastAPI not available
    pass
