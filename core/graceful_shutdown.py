#!/usr/bin/env python3
"""
Graceful Shutdown Manager for TORI
==================================

Handles clean shutdown of all TORI services when receiving interrupt signals.
"""

import signal
import sys
import asyncio
import threading
import time
import logging
from typing import List, Callable, Any
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tori.shutdown')

class GracefulShutdownManager:
    """Manages graceful shutdown of TORI services"""
    
    def __init__(self):
        self.shutdown_handlers: List[Callable] = []
        self.is_shutting_down = False
        self.shutdown_event = threading.Event()
        self.async_shutdown_event = None
        self._original_sigint = None
        self._original_sigterm = None
        
    def register_handler(self, handler: Callable, name: str = None):
        """Register a shutdown handler function"""
        handler_name = name or handler.__name__
        logger.info(f"Registered shutdown handler: {handler_name}")
        self.shutdown_handlers.append((handler, handler_name))
        
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested"""
        return self.is_shutting_down
    
    def get_shutdown_event(self) -> threading.Event:
        """Get the shutdown event for thread synchronization"""
        return self.shutdown_event
    
    def get_async_shutdown_event(self) -> asyncio.Event:
        """Get the async shutdown event for asyncio synchronization"""
        if self.async_shutdown_event is None:
            self.async_shutdown_event = asyncio.Event()
        return self.async_shutdown_event
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        if self.is_shutting_down:
            logger.warning("Force shutdown requested - terminating immediately!")
            sys.exit(1)
            
        logger.info(f"\n🛑 Shutdown signal received (signal {signum})")
        self.is_shutting_down = True
        self.shutdown_event.set()
        
        # Set async event if it exists
        if self.async_shutdown_event:
            asyncio.create_task(self._set_async_event())
        
        # Execute shutdown handlers
        self._execute_shutdown()
        
    async def _set_async_event(self):
        """Set the async shutdown event"""
        self.async_shutdown_event.set()
        
    def _execute_shutdown(self):
        """Execute all registered shutdown handlers"""
        logger.info("🔄 Starting graceful shutdown sequence...")
        
        # Execute handlers in reverse order (LIFO)
        for handler, name in reversed(self.shutdown_handlers):
            try:
                logger.info(f"  ⏸️  Stopping {name}...")
                result = handler()
                
                # Handle async handlers
                if asyncio.iscoroutine(result):
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(result)
                    
                logger.info(f"  ✅ {name} stopped successfully")
            except Exception as e:
                logger.error(f"  ❌ Error stopping {name}: {e}")
        
        logger.info("✅ Graceful shutdown complete")
        
        # Restore original signal handlers
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm:
            signal.signal(signal.SIGTERM, self._original_sigterm)
            
    def install_signal_handlers(self):
        """Install signal handlers for graceful shutdown"""
        # Store original handlers
        self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        logger.info("✅ Graceful shutdown handlers installed")

# Global instance
shutdown_manager = GracefulShutdownManager()

def register_shutdown_handler(handler: Callable, name: str = None):
    """Register a shutdown handler with the global manager"""
    shutdown_manager.register_handler(handler, name)

def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested"""
    return shutdown_manager.is_shutdown_requested()

def get_shutdown_event() -> threading.Event:
    """Get the shutdown event for thread synchronization"""
    return shutdown_manager.get_shutdown_event()

def install_shutdown_handlers():
    """Install signal handlers for graceful shutdown"""
    shutdown_manager.install_signal_handlers()

# Example shutdown handlers for TORI services
def create_lattice_shutdown_handler(lattice_runner):
    """Create a shutdown handler for the lattice evolution runner"""
    def handler():
        logger.info("Stopping lattice evolution...")
        if hasattr(lattice_runner, 'stop'):
            lattice_runner.stop()
        elif hasattr(lattice_runner, 'shutdown'):
            lattice_runner.shutdown()
    return handler

def create_mcp_shutdown_handler(mcp_process):
    """Create a shutdown handler for the MCP server"""
    def handler():
        logger.info("Stopping MCP server...")
        if mcp_process and mcp_process.poll() is None:
            mcp_process.terminate()
            try:
                mcp_process.wait(timeout=5)
            except:
                mcp_process.kill()
    return handler

def create_api_shutdown_handler(api_server):
    """Create a shutdown handler for the API server"""
    async def handler():
        logger.info("Stopping API server...")
        if hasattr(api_server, 'shutdown'):
            await api_server.shutdown()
    return handler

def create_frontend_shutdown_handler(frontend_process):
    """Create a shutdown handler for the frontend"""
    def handler():
        logger.info("Stopping frontend...")
        if frontend_process and frontend_process.poll() is None:
            frontend_process.terminate()
            try:
                frontend_process.wait(timeout=5)
            except:
                frontend_process.kill()
    return handler

if __name__ == "__main__":
    # Test the shutdown manager
    print("Testing graceful shutdown manager...")
    
    def test_handler():
        print("Test handler executing...")
        time.sleep(1)
        print("Test handler complete")
    
    register_shutdown_handler(test_handler, "Test Service")
    install_shutdown_handlers()
    
    print("Press Ctrl+C to test shutdown...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
