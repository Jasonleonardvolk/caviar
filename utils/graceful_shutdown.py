"""
Graceful Shutdown Handler for TORI System
Manages clean shutdown of all services and resources
"""

import signal
import asyncio
import sys
import logging
from typing import Optional, Callable, List, Set
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)


class GracefulShutdownHandler:
    """Handles graceful shutdown of the system"""
    
    def __init__(self):
        self.shutdown_initiated = False
        self.cleanup_callbacks: List[Callable] = []
        self._original_sigint = None
        self._original_sigterm = None
        
    def register_cleanup(self, callback: Callable):
        """Register a cleanup callback to run on shutdown"""
        self.cleanup_callbacks.append(callback)
        
    def add_cleanup_callback(self, callback: Callable):
        """Register a cleanup callback (alias for register_cleanup)"""
        self.register_cleanup(callback)
        
    def register_process(self, process, name: str = "process", is_critical: bool = False):
        """Register a process for cleanup on shutdown"""
        def cleanup_process():
            try:
                if hasattr(process, 'terminate'):
                    process.terminate()
                    logger.info(f"Terminated {name} process")
            except Exception as e:
                logger.error(f"Error terminating {name} process: {e}")
        
        self.register_cleanup(cleanup_process)
        
        # Store critical status for potential future use
        if is_critical:
            logger.info(f"Registered critical process: {name}")
        
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signal"""
        if self.shutdown_initiated:
            logger.warning("Forced shutdown requested - exiting immediately")
            sys.exit(1)
            
        logger.info(f"Graceful shutdown initiated (signal {signum})")
        self.shutdown_initiated = True
        
        # Run cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback: {e}")
                
        # Exit gracefully
        sys.exit(0)
        
    def setup_signal_handlers(self):
        """Setup signal handlers (alias for setup method)"""
        self.setup()
        
    def setup(self):
        """Setup signal handlers"""
        self._original_sigint = signal.signal(signal.SIGINT, self.handle_shutdown)
        self._original_sigterm = signal.signal(signal.SIGTERM, self.handle_shutdown)
        
    def restore(self):
        """Restore original signal handlers"""
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm:
            signal.signal(signal.SIGTERM, self._original_sigterm)


class AsyncioGracefulShutdown:
    """Asyncio-compatible graceful shutdown handler"""
    
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.tasks: Set[asyncio.Task] = set()
        self.cleanup_coros: List[Callable] = []
        
    def register_cleanup(self, coro: Callable):
        """Register an async cleanup coroutine"""
        self.cleanup_coros.append(coro)
        
    def track_task(self, task: asyncio.Task):
        """Track a task for cleanup"""
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        
    async def shutdown(self):
        """Perform async shutdown"""
        logger.info("Starting async shutdown sequence")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Run cleanup coroutines
        for coro in self.cleanup_coros:
            try:
                await coro()
            except Exception as e:
                logger.error(f"Error in async cleanup: {e}")
                
        # Cancel remaining tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        logger.info("Async shutdown complete")
        
    def setup_signal_handlers(self):
        """Setup asyncio signal handlers"""
        loop = asyncio.get_event_loop()
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig, 
                lambda: asyncio.create_task(self.shutdown())
            )


@contextmanager
def delayed_keyboard_interrupt():
    """Context manager to delay keyboard interrupts"""
    old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old_handler)


# Global handlers
graceful_handler = GracefulShutdownHandler()
async_handler = AsyncioGracefulShutdown()


def setup_graceful_shutdown():
    """Setup both sync and async shutdown handlers"""
    graceful_handler.setup()
    try:
        async_handler.setup_signal_handlers()
    except RuntimeError:
        # No event loop yet, will be set up later
        pass
