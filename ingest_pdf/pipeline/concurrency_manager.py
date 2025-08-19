"""
concurrency_manager.py

Dedicated concurrency management with separate executors for CPU and I/O tasks.
Provides clean async/sync abstraction layer.
"""

import asyncio
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from typing import Optional, Callable, Any, TypeVar, Union, Dict, List, Awaitable, Coroutine
from dataclasses import dataclass, field
from functools import wraps
import psutil

logger = logging.getLogger("tori.ingest_pdf.concurrency")

T = TypeVar('T')

@dataclass
class ExecutorStats:
    """Statistics for executor monitoring."""
    tasks_submitted: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_duration: float = 0.0
    average_duration: float = 0.0
    active_tasks: int = 0
    max_concurrent: int = 0

@dataclass 
class ConcurrencyConfig:
    """Configuration for concurrency management."""
    cpu_workers: Optional[int] = None
    io_workers: int = 20
    chunk_processor_workers: Optional[int] = None
    enable_auto_throttle: bool = True
    cpu_threshold: float = 85.0  # CPU usage percentage threshold
    memory_threshold: float = 80.0  # Memory usage percentage threshold
    max_queue_size: int = 1000
    enable_metrics: bool = True
    
    def __post_init__(self):
        if self.cpu_workers is None:
            # Default to CPU count minus 1 to leave room for main thread
            self.cpu_workers = min(8, max(1, (os.cpu_count() or 4) - 1))
        if self.chunk_processor_workers is None:
            # For chunk processing, use more workers
            self.chunk_processor_workers = min(8, os.cpu_count() or 1)

class ConcurrencyManager:
    """
    Centralized concurrency management with dedicated executors.
    
    Features:
    - Separate executors for CPU-bound and I/O-bound tasks
    - Clean async/sync abstraction layer
    - Auto-throttling based on system resources
    - Task batching and fan-out
    - Metrics and monitoring
    - Graceful shutdown
    """
    
    def __init__(self, config: Optional[ConcurrencyConfig] = None):
        self.config = config or ConcurrencyConfig()
        
        # Create dedicated executors
        self.cpu_executor = ProcessPoolExecutor(
            max_workers=self.config.cpu_workers,
            initializer=self._process_initializer
        )
        
        self.io_executor = ThreadPoolExecutor(
            max_workers=self.config.io_workers,
            thread_name_prefix="io_worker"
        )
        
        # Separate executor for chunk processing with higher concurrency
        self.chunk_executor = ThreadPoolExecutor(
            max_workers=self.config.chunk_processor_workers,
            thread_name_prefix="chunk_worker"
        )
        
        # Executor statistics
        self.stats: Dict[str, ExecutorStats] = {
            'cpu': ExecutorStats(),
            'io': ExecutorStats(),
            'chunk': ExecutorStats()
        }
        
        # Semaphores for throttling
        self._cpu_semaphore = asyncio.Semaphore(self.config.cpu_workers)
        self._io_semaphore = asyncio.Semaphore(self.config.io_workers)
        self._chunk_semaphore = asyncio.Semaphore(self.config.chunk_processor_workers)
        
        # Auto-throttle state
        self._throttle_factor = 1.0
        self._last_throttle_check = 0.0
        
        logger.info(f"ConcurrencyManager initialized with {self.config.cpu_workers} CPU workers, "
                   f"{self.config.io_workers} I/O workers, {self.config.chunk_processor_workers} chunk workers")
    
    @staticmethod
    def _process_initializer():
        """Initialize process pool workers."""
        # Set up any process-specific initialization
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    async def run_cpu(self, fn: Callable[..., T], *args, **kwargs) -> T:
        """Run CPU-bound function in process pool."""
        return await self._run_in_executor('cpu', self.cpu_executor, self._cpu_semaphore, fn, *args, **kwargs)
    
    async def run_io(self, fn: Callable[..., T], *args, **kwargs) -> T:
        """Run I/O-bound function in thread pool."""
        return await self._run_in_executor('io', self.io_executor, self._io_semaphore, fn, *args, **kwargs)
    
    async def run_chunk(self, fn: Callable[..., T], *args, **kwargs) -> T:
        """Run chunk processing in dedicated thread pool."""
        return await self._run_in_executor('chunk', self.chunk_executor, self._chunk_semaphore, fn, *args, **kwargs)
    
    async def _run_in_executor(self, executor_name: str, executor, semaphore: asyncio.Semaphore, 
                              fn: Callable[..., T], *args, **kwargs) -> T:
        """Internal method to run function in executor with monitoring."""
        stats = self.stats[executor_name]
        
        # Check throttling
        if self.config.enable_auto_throttle:
            await self._check_throttle()
        
        async with semaphore:
            start_time = time.time()
            stats.tasks_submitted += 1
            stats.active_tasks += 1
            stats.max_concurrent = max(stats.max_concurrent, stats.active_tasks)
            
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(executor, fn, *args)
                stats.tasks_completed += 1
                return result
            except Exception as e:
                stats.tasks_failed += 1
                logger.error(f"Error in {executor_name} executor: {e}")
                raise
            finally:
                duration = time.time() - start_time
                stats.total_duration += duration
                stats.active_tasks -= 1
                if stats.tasks_completed > 0:
                    stats.average_duration = stats.total_duration / stats.tasks_completed
    
    async def _check_throttle(self):
        """Check system resources and adjust throttling."""
        current_time = time.time()
        if current_time - self._last_throttle_check < 1.0:  # Check at most once per second
            return
        
        self._last_throttle_check = current_time
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            if cpu_percent > self.config.cpu_threshold or memory_percent > self.config.memory_threshold:
                # Reduce concurrency
                self._throttle_factor = max(0.5, self._throttle_factor * 0.9)
                logger.warning(f"System under load (CPU: {cpu_percent}%, Memory: {memory_percent}%), "
                             f"throttling to {self._throttle_factor:.1%}")
            else:
                # Gradually restore concurrency
                self._throttle_factor = min(1.0, self._throttle_factor * 1.05)
        except Exception as e:
            logger.debug(f"Error checking system resources: {e}")
    
    async def map_cpu(self, fn: Callable[..., T], items: List[Any], *args, **kwargs) -> List[T]:
        """Map CPU-bound function over items with batching."""
        tasks = [self.run_cpu(fn, item, *args, **kwargs) for item in items]
        return await asyncio.gather(*tasks)
    
    async def map_io(self, fn: Callable[..., T], items: List[Any], *args, **kwargs) -> List[T]:
        """Map I/O-bound function over items with batching."""
        tasks = [self.run_io(fn, item, *args, **kwargs) for item in items]
        return await asyncio.gather(*tasks)
    
    async def map_chunks(self, fn: Callable[..., T], chunks: List[Any], *args, **kwargs) -> List[T]:
        """Map chunk processing function over chunks with dedicated executor."""
        tasks = [self.run_chunk(fn, chunk, *args, **kwargs) for chunk in chunks]
        return await asyncio.gather(*tasks)
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get executor statistics."""
        return {
            name: {
                'tasks_submitted': stats.tasks_submitted,
                'tasks_completed': stats.tasks_completed,
                'tasks_failed': stats.tasks_failed,
                'average_duration': round(stats.average_duration, 3),
                'active_tasks': stats.active_tasks,
                'max_concurrent': stats.max_concurrent,
                'success_rate': round(stats.tasks_completed / max(1, stats.tasks_submitted) * 100, 1)
            }
            for name, stats in self.stats.items()
        }
    
    def shutdown(self, wait: bool = True):
        """Shutdown all executors gracefully."""
        logger.info("Shutting down ConcurrencyManager...")
        
        # Log final stats
        if self.config.enable_metrics:
            stats = self.get_stats()
            for name, stat in stats.items():
                logger.info(f"{name} executor stats: {stat}")
        
        # Shutdown executors
        self.cpu_executor.shutdown(wait=wait)
        self.io_executor.shutdown(wait=wait) 
        self.chunk_executor.shutdown(wait=wait)
        
        logger.info("ConcurrencyManager shutdown complete")

# Global instance (created on first access)
_global_manager: Optional[ConcurrencyManager] = None

def get_concurrency_manager(config: Optional[ConcurrencyConfig] = None) -> ConcurrencyManager:
    """Get or create global concurrency manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = ConcurrencyManager(config)
    return _global_manager

# Convenience decorators
def run_cpu_bound(fn: Callable[..., T]) -> Callable[..., asyncio.Future[T]]:
    """Decorator to automatically run function in CPU executor."""
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        manager = get_concurrency_manager()
        return await manager.run_cpu(fn, *args, **kwargs)
    return wrapper

def run_io_bound(fn: Callable[..., T]) -> Callable[..., asyncio.Future[T]]:
    """Decorator to automatically run function in I/O executor."""
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        manager = get_concurrency_manager()
        return await manager.run_io(fn, *args, **kwargs)
    return wrapper

# Unified async/sync adapter
async def run_sync(sync_func: Callable[..., T], *args, 
                  executor_type: str = 'io', **kwargs) -> T:
    """
    Run synchronous function in appropriate executor.
    
    Args:
        sync_func: Synchronous function to run
        executor_type: 'cpu', 'io', or 'chunk'
        *args, **kwargs: Arguments for the function
        
    Returns:
        Function result
    """
    manager = get_concurrency_manager()
    
    if executor_type == 'cpu':
        return await manager.run_cpu(sync_func, *args, **kwargs)
    elif executor_type == 'chunk':
        return await manager.run_chunk(sync_func, *args, **kwargs)
    else:
        return await manager.run_io(sync_func, *args, **kwargs)

# Batch processing utilities
async def process_in_batches(items: List[Any], 
                           process_fn: Callable[[Any], T],
                           batch_size: int = 10,
                           executor_type: str = 'cpu') -> List[T]:
    """
    Process items in batches using appropriate executor.
    
    Args:
        items: Items to process
        process_fn: Function to process each item
        batch_size: Number of items per batch
        executor_type: Executor to use
        
    Returns:
        List of results
    """
    manager = get_concurrency_manager()
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        if executor_type == 'cpu':
            batch_results = await manager.map_cpu(process_fn, batch)
        elif executor_type == 'chunk':
            batch_results = await manager.map_chunks(process_fn, batch)
        else:
            batch_results = await manager.map_io(process_fn, batch)
            
        results.extend(batch_results)
        
        # Yield control periodically
        if i % (batch_size * 10) == 0:
            await asyncio.sleep(0)
    
    return results

# Export main components
__all__ = [
    'ConcurrencyManager',
    'ConcurrencyConfig',
    'ExecutorStats',
    'get_concurrency_manager',
    'run_cpu_bound',
    'run_io_bound',
    'run_sync',
    'process_in_batches'
]
