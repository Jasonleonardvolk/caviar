"""
Execution helpers for async/sync interoperability in TORI Pipeline.
Provides efficient thread management and event loop reuse.
"""

import asyncio
import atexit
import concurrent.futures
import threading
from typing import Any, Coroutine, TypeVar, Optional, Union
import logging
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')

class PersistentLoopExecutor:
    """
    Thread executor with persistent event loop to avoid recreation overhead.
    Runs a single event loop in a dedicated thread for all async operations.
    """
    
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._executor: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="tori-async"
        )
        self._started = threading.Event()
        self._lock = threading.Lock()
        
    def _ensure_loop(self):
        """Ensure the event loop is running in the executor thread."""
        with self._lock:
            if self._loop is None or not self._loop.is_running():
                self._started.clear()
                future = self._executor.submit(self._run_loop)
                # Wait for loop to start
                self._started.wait(timeout=5.0)
                if not self._started.is_set():
                    raise RuntimeError("Failed to start event loop in executor")
                    
    def _run_loop(self):
        """Run event loop in executor thread."""
        try:
            # Create new event loop for this thread
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            
            # Signal that loop is ready
            self._started.set()
            
            # Run forever until shutdown
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Event loop error: {e}")
        finally:
            if self._loop:
                self._loop.close()
                self._loop = None
                
    def run_coroutine(self, coro: Coroutine[Any, Any, T], submit_only: bool = False) -> Union[T, concurrent.futures.Future]:
        """
        Run a coroutine in the persistent event loop.
        
        Args:
            coro: The coroutine to run
            submit_only: If True, return the Future instead of waiting for result
            
        Returns:
            The coroutine result, or a Future if submit_only=True
        """
        # Ensure loop is running
        self._ensure_loop()
        
        if not self._loop:
            raise RuntimeError("Event loop not available")
            
        # Schedule coroutine in the loop
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        
        if submit_only:
            return future
            
        # Wait for result
        return future.result()
        
    def shutdown(self, wait: bool = True):
        """Shutdown the executor and event loop cleanly."""
        with self._lock:
            # Stop the event loop
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
                
            # Shutdown executor
            self._executor.shutdown(wait=wait, cancel_futures=True)
            
            # Clear references
            self._loop = None


# Global persistent loop executor
_persistent_executor = PersistentLoopExecutor()

# Register cleanup on exit
atexit.register(lambda: _persistent_executor.shutdown(wait=True))


def run_sync(coro: Coroutine[Any, Any, T], *, timeout: Optional[float] = None) -> T:
    """
    Run a coroutine synchronously from any context.
    
    Uses a persistent event loop in a dedicated thread to avoid the overhead
    of creating a new loop for each call. This is significantly more efficient
    for applications that make many async calls from sync contexts.
    
    When called from async context: Uses run_coroutine_threadsafe on the current loop.
    When called from sync context: Uses the persistent loop in the executor thread.
    
    Args:
        coro: The coroutine to run
        timeout: Optional timeout in seconds. Raises concurrent.futures.TimeoutError if exceeded.
    
    Returns:
        The result of the coroutine
        
    Raises:
        concurrent.futures.TimeoutError: If timeout is exceeded
        Exception: Any exception raised by the coroutine
    
    WARNING: This blocks the calling thread. For non-blocking behavior,
    use 'await' directly instead of run_sync().
    
    NOTE: When called inside FastAPI request handlers, this will block
    that worker thread, potentially serializing requests on that worker.
    Consider using async handlers with 'await' instead.
    """
    future = None
    
    try:
        # Check if we're in an async context
        loop = asyncio.get_running_loop()
        
        # We're in async context - use run_coroutine_threadsafe
        # This blocks the caller thread but lets the event loop continue
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        
    except RuntimeError:
        # No loop running - use persistent executor
        future = _persistent_executor.run_coroutine(coro, submit_only=True)
    
    # Get result with timeout
    try:
        return future.result(timeout=timeout)
    except Exception:
        # Attempt to cancel on any exception
        if future and not future.done():
            future.cancel()
            # Log cancellation attempt
            if future.cancelled():
                logger.debug("run_sync: Successfully cancelled coroutine after exception")
            else:
                logger.debug("run_sync: Could not cancel coroutine (may have completed)")
        raise
    finally:
        # Log if the future was already cancelled
        if future and future.cancelled():
            logger.debug("run_sync: Coroutine was cancelled")


async def await_sync(coro: Coroutine[Any, Any, T]) -> T:
    """
    Helper for truly non-blocking execution in async contexts.
    
    Unlike run_sync which blocks the calling thread, this returns
    immediately and can be awaited.
    """
    return await coro


def shutdown_executors():
    """
    Manually shutdown all executors.
    
    This is automatically called on exit, but can be called
    manually for testing or graceful shutdown scenarios.
    """
    _persistent_executor.shutdown(wait=True)
    logger.info("Execution helpers shutdown complete")


# For backwards compatibility
__all__ = ['run_sync', 'await_sync', 'shutdown_executors']
