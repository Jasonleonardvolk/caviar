#!/usr/bin/env python3
"""
GPU-Accelerated Eigenvalue Monitor for TORI/KHA
Implements CUDA acceleration for eigenvalue computations with fallback to CPU
No containers or databases - pure file-based operation
"""

import numpy as np
import time
import logging
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from functools import lru_cache
import hashlib
import pickle
import json

# Try to import PyTorch for GPU support
try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available - GPU acceleration disabled")

# Try to import CuPy as alternative
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logging.warning("CuPy not available - Alternative GPU acceleration disabled")

from python.stability.eigenvalue_monitor import EigenvalueMonitor, EigenvalueAnalysis

logger = logging.getLogger(__name__)

class GPUEigenvalueMonitor(EigenvalueMonitor):
    """
    GPU-accelerated eigenvalue monitor with intelligent caching
    Falls back to CPU when GPU not available
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # GPU configuration
        self.use_gpu = self.config.get('use_gpu', True)
        self.gpu_backend = self._detect_gpu_backend()
        self.device = None
        
        # Initialize GPU if available
        if self.gpu_backend:
            self._initialize_gpu()
            
        # LRU cache configuration
        self.cache_size = self.config.get('cache_size', 1000)
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour
        
        # File-based cache for persistence
        self.cache_dir = self.storage_path / 'eigenvalue_cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        # Performance metrics
        self.gpu_computations = 0
        self.cpu_computations = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"GPUEigenvalueMonitor initialized with backend: {self.gpu_backend}")
    
    def _detect_gpu_backend(self) -> Optional[str]:
        """Detect available GPU backend"""
        if not self.use_gpu:
            return None
            
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return 'pytorch'
        elif CUPY_AVAILABLE:
            try:
                # Test CuPy GPU availability
                cp.cuda.runtime.getDeviceCount()
                return 'cupy'
            except:
                pass
                
        return None
    
    def _initialize_gpu(self):
        """Initialize GPU device"""
        if self.gpu_backend == 'pytorch':
            self.device = torch.device('cuda')
            # Warm up GPU
            test_tensor = torch.randn(10, 10, device=self.device)
            torch.linalg.eig(test_tensor)
            logger.info(f"PyTorch GPU initialized: {torch.cuda.get_device_name()}")
            
        elif self.gpu_backend == 'cupy':
            self.device = cp.cuda.Device(0)
            # Warm up GPU
            with self.device:
                test_array = cp.random.randn(10, 10)
                cp.linalg.eig(test_array)
            logger.info(f"CuPy GPU initialized: Device {self.device.id}")
    
    @lru_cache(maxsize=1000)
    def _compute_matrix_hash(self, matrix_bytes: bytes) -> str:
        """Compute hash of matrix for caching"""
        return hashlib.sha256(matrix_bytes).hexdigest()
    
    def _get_cached_result(self, matrix_hash: str) -> Optional[EigenvalueAnalysis]:
        """Retrieve cached eigenvalue analysis"""
        # Check memory cache first (handled by lru_cache)
        
        # Check file cache
        cache_file = self.cache_dir / f"{matrix_hash}.pkl"
        if cache_file.exists():
            try:
                # Check if cache is still valid
                cache_age = time.time() - cache_file.stat().st_mtime
                if cache_age < self.cache_ttl:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    # Reconstruct EigenvalueAnalysis
                    analysis = EigenvalueAnalysis(**cached_data)
                    self.cache_hits += 1
                    logger.debug(f"Cache hit for matrix hash: {matrix_hash[:8]}...")
                    return analysis
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        self.cache_misses += 1
        return None
    
    def _save_to_cache(self, matrix_hash: str, analysis: EigenvalueAnalysis):
        """Save analysis to file cache"""
        cache_file = self.cache_dir / f"{matrix_hash}.pkl"
        try:
            # Convert to dict for pickling
            cache_data = {
                'eigenvalues': analysis.eigenvalues,
                'eigenvectors': analysis.eigenvectors,
                'max_eigenvalue': analysis.max_eigenvalue,
                'spectral_radius': analysis.spectral_radius,
                'condition_number': analysis.condition_number,
                'numerical_rank': analysis.numerical_rank,
                'stability_margin': analysis.stability_margin,
                'is_stable': analysis.is_stable,
                'is_hermitian': analysis.is_hermitian,
                'is_normal': analysis.is_normal,
                'timestamp': analysis.timestamp
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    async def analyze_matrix(self, matrix: np.ndarray) -> EigenvalueAnalysis:
        """
        GPU-accelerated eigenvalue analysis with caching
        """
        # Compute matrix hash for caching
        matrix_bytes = matrix.tobytes()
        matrix_hash = self._compute_matrix_hash(matrix_bytes)
        
        # Check cache first
        cached_result = self._get_cached_result(matrix_hash)
        if cached_result is not None:
            return cached_result
        
        # Perform GPU or CPU computation
        if self.gpu_backend:
            try:
                analysis = await self._analyze_matrix_gpu(matrix)
                self.gpu_computations += 1
            except Exception as e:
                logger.warning(f"GPU computation failed, falling back to CPU: {e}")
                analysis = await super().analyze_matrix(matrix)
                self.cpu_computations += 1
        else:
            analysis = await super().analyze_matrix(matrix)
            self.cpu_computations += 1
        
        # Save to cache
        self._save_to_cache(matrix_hash, analysis)
        
        return analysis
    
    async def _analyze_matrix_gpu(self, matrix: np.ndarray) -> EigenvalueAnalysis:
        """GPU-accelerated eigenvalue computation"""
        start_time = time.time()
        
        with self.lock:
            # Validate input
            if not self._validate_matrix(matrix):
                raise ValueError("Invalid matrix input")
            
            # Check matrix properties
            is_hermitian = np.allclose(matrix, matrix.conj().T)
            is_normal = np.allclose(matrix @ matrix.conj().T, matrix.conj().T @ matrix)
            
            if self.gpu_backend == 'pytorch':
                # PyTorch GPU computation
                matrix_gpu = torch.from_numpy(matrix).to(self.device, dtype=torch.complex64)
                
                if is_hermitian:
                    eigenvalues_gpu, eigenvectors_gpu = torch.linalg.eigh(matrix_gpu)
                else:
                    eigenvalues_gpu, eigenvectors_gpu = torch.linalg.eig(matrix_gpu)
                
                # Transfer back to CPU
                eigenvalues = eigenvalues_gpu.cpu().numpy()
                eigenvectors = eigenvectors_gpu.cpu().numpy()
                
            elif self.gpu_backend == 'cupy':
                # CuPy GPU computation
                with self.device:
                    matrix_gpu = cp.asarray(matrix, dtype=cp.complex64)
                    
                    if is_hermitian:
                        eigenvalues_gpu, eigenvectors_gpu = cp.linalg.eigh(matrix_gpu)
                    else:
                        eigenvalues_gpu, eigenvectors_gpu = cp.linalg.eig(matrix_gpu)
                    
                    # Transfer back to CPU
                    eigenvalues = cp.asnumpy(eigenvalues_gpu)
                    eigenvectors = cp.asnumpy(eigenvectors_gpu)
            
            # Sort by magnitude
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Compute metrics (same as parent class)
            max_eigenvalue = np.max(np.abs(eigenvalues))
            spectral_radius = max_eigenvalue
            
            # Condition number using GPU if possible
            if self.gpu_backend == 'pytorch':
                cond = torch.linalg.cond(matrix_gpu).cpu().item()
                condition_number = float(cond)
            elif self.gpu_backend == 'cupy':
                with self.device:
                    condition_number = float(cp.linalg.cond(matrix_gpu))
            else:
                condition_number = np.linalg.cond(matrix)
            
            # Numerical rank
            if self.gpu_backend == 'pytorch':
                rank = torch.linalg.matrix_rank(matrix_gpu).cpu().item()
                numerical_rank = int(rank)
            elif self.gpu_backend == 'cupy':
                with self.device:
                    numerical_rank = int(cp.linalg.matrix_rank(matrix_gpu))
            else:
                _, s, _ = np.linalg.svd(matrix)
                numerical_rank = np.sum(s > 1e-10 * s[0])
            
            # Stability calculations
            stability_margin = self.stability_threshold - max_eigenvalue
            is_stable = max_eigenvalue < self.stability_threshold
            
            # Create analysis result
            analysis = EigenvalueAnalysis(
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                max_eigenvalue=float(max_eigenvalue),
                spectral_radius=float(spectral_radius),
                condition_number=float(condition_number),
                numerical_rank=int(numerical_rank),
                stability_margin=float(stability_margin),
                is_stable=bool(is_stable),
                is_hermitian=bool(is_hermitian),
                is_normal=bool(is_normal),
                timestamp=time.time()
            )
            
            # Update history
            self.eigenvalue_history.append(analysis)
            self.matrix_history.append(matrix.copy())
            
            # Check for warnings
            await self._check_warnings(analysis)
            
            # Notify callbacks if unstable
            if not is_stable:
                await self._notify_instability(analysis)
            
            gpu_time = time.time() - start_time
            logger.debug(f"GPU eigenvalue analysis completed in {gpu_time:.3f}s")
            
            return analysis
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get GPU performance metrics"""
        total_computations = self.gpu_computations + self.cpu_computations
        
        metrics = {
            'gpu_backend': self.gpu_backend or 'None',
            'gpu_computations': self.gpu_computations,
            'cpu_computations': self.cpu_computations,
            'gpu_percentage': (self.gpu_computations / total_computations * 100) if total_computations > 0 else 0,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': (self.cache_hits / (self.cache_hits + self.cache_misses) * 100) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'cache_size_mb': sum(f.stat().st_size for f in self.cache_dir.glob('*.pkl')) / (1024 * 1024)
        }
        
        # Add GPU memory info if available
        if self.gpu_backend == 'pytorch' and torch.cuda.is_available():
            metrics['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
            metrics['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
        elif self.gpu_backend == 'cupy':
            with self.device:
                mempool = cp.get_default_memory_pool()
                metrics['gpu_memory_used_mb'] = mempool.used_bytes() / (1024 * 1024)
                metrics['gpu_memory_total_mb'] = mempool.total_bytes() / (1024 * 1024)
        
        return metrics
    
    def clear_cache(self):
        """Clear eigenvalue cache"""
        # Clear file cache
        cache_files = list(self.cache_dir.glob('*.pkl'))
        for cache_file in cache_files:
            try:
                cache_file.unlink()
            except:
                pass
        
        # Clear LRU cache
        self._compute_matrix_hash.cache_clear()
        
        # Reset counters
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"Cleared {len(cache_files)} cached eigenvalue results")
    
    def optimize_cache(self):
        """Remove old cache entries to save space"""
        current_time = time.time()
        removed = 0
        
        for cache_file in self.cache_dir.glob('*.pkl'):
            try:
                age = current_time - cache_file.stat().st_mtime
                if age > self.cache_ttl:
                    cache_file.unlink()
                    removed += 1
            except:
                pass
        
        logger.info(f"Removed {removed} expired cache entries")
    
    async def batch_analyze_matrices(self, matrices: List[np.ndarray]) -> List[EigenvalueAnalysis]:
        """
        Analyze multiple matrices in batch for better GPU utilization
        """
        if not self.gpu_backend or len(matrices) == 1:
            # Fall back to sequential processing
            results = []
            for matrix in matrices:
                result = await self.analyze_matrix(matrix)
                results.append(result)
            return results
        
        # Batch GPU processing
        logger.info(f"Batch analyzing {len(matrices)} matrices on GPU")
        
        # Group matrices by size for efficient batching
        size_groups = {}
        for i, matrix in enumerate(matrices):
            size = matrix.shape[0]
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append((i, matrix))
        
        results = [None] * len(matrices)
        
        for size, group in size_groups.items():
            indices = [g[0] for g in group]
            matrices_group = [g[1] for g in group]
            
            # Process group on GPU
            if self.gpu_backend == 'pytorch':
                # Stack matrices for batch processing
                batch_tensor = torch.stack([torch.from_numpy(m) for m in matrices_group]).to(self.device)
                
                # Batch eigenvalue computation
                eigenvalues_batch, eigenvectors_batch = torch.linalg.eig(batch_tensor)
                
                # Process results
                for i, idx in enumerate(indices):
                    eigenvalues = eigenvalues_batch[i].cpu().numpy()
                    eigenvectors = eigenvectors_batch[i].cpu().numpy()
                    
                    # Create analysis (simplified for batch)
                    analysis = self._create_analysis_from_eigendata(
                        matrices_group[i], eigenvalues, eigenvectors
                    )
                    results[idx] = analysis
                    
            elif self.gpu_backend == 'cupy':
                with self.device:
                    for i, (idx, matrix) in enumerate(group):
                        matrix_gpu = cp.asarray(matrix)
                        eigenvalues, eigenvectors = cp.linalg.eig(matrix_gpu)
                        
                        analysis = self._create_analysis_from_eigendata(
                            matrix, cp.asnumpy(eigenvalues), cp.asnumpy(eigenvectors)
                        )
                        results[idx] = analysis
        
        return results
    
    def _create_analysis_from_eigendata(self, matrix: np.ndarray, 
                                      eigenvalues: np.ndarray, 
                                      eigenvectors: np.ndarray) -> EigenvalueAnalysis:
        """Create EigenvalueAnalysis from computed eigendata"""
        # Sort by magnitude
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        max_eigenvalue = np.max(np.abs(eigenvalues))
        
        return EigenvalueAnalysis(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            max_eigenvalue=float(max_eigenvalue),
            spectral_radius=float(max_eigenvalue),
            condition_number=float(np.linalg.cond(matrix)),
            numerical_rank=int(np.linalg.matrix_rank(matrix)),
            stability_margin=float(self.stability_threshold - max_eigenvalue),
            is_stable=bool(max_eigenvalue < self.stability_threshold),
            is_hermitian=bool(np.allclose(matrix, matrix.conj().T)),
            is_normal=bool(np.allclose(matrix @ matrix.conj().T, matrix.conj().T @ matrix)),
            timestamp=time.time()
        )
    
    def save_performance_report(self):
        """Save performance metrics to file"""
        report = {
            'timestamp': time.time(),
            'performance_metrics': self.get_performance_metrics(),
            'cache_stats': {
                'total_cached': len(list(self.cache_dir.glob('*.pkl'))),
                'cache_directory': str(self.cache_dir)
            }
        }
        
        report_file = self.storage_path / 'gpu_performance_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to {report_file}")

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_gpu_eigenvalue_monitor():
        """Test GPU-accelerated eigenvalue monitor"""
        config = {
            'matrix_size': 100,
            'use_gpu': True,
            'cache_size': 500,
            'storage_path': 'data/gpu_eigenvalue_test'
        }
        
        monitor = GPUEigenvalueMonitor(config)
        
        print(f"GPU Backend: {monitor.gpu_backend}")
        
        # Test single matrix
        test_matrix = np.random.randn(50, 50) + 1j * np.random.randn(50, 50)
        
        print("\nTesting single matrix analysis...")
        start = time.time()
        analysis1 = await monitor.analyze_matrix(test_matrix)
        gpu_time = time.time() - start
        print(f"First analysis (GPU): {gpu_time:.3f}s")
        print(f"Max eigenvalue: {analysis1.max_eigenvalue:.4f}")
        
        # Test cache hit
        start = time.time()
        analysis2 = await monitor.analyze_matrix(test_matrix)
        cache_time = time.time() - start
        print(f"Second analysis (cached): {cache_time:.3f}s")
        print(f"Speedup: {gpu_time/cache_time:.1f}x")
        
        # Test batch processing
        print("\nTesting batch processing...")
        matrices = [np.random.randn(30, 30) for _ in range(10)]
        start = time.time()
        results = await monitor.batch_analyze_matrices(matrices)
        batch_time = time.time() - start
        print(f"Batch analysis of {len(matrices)} matrices: {batch_time:.3f}s")
        print(f"Average time per matrix: {batch_time/len(matrices):.3f}s")
        
        # Show performance metrics
        print("\nPerformance Metrics:")
        metrics = monitor.get_performance_metrics()
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Save report
        monitor.save_performance_report()
        monitor.shutdown()
    
    asyncio.run(test_gpu_eigenvalue_monitor())
