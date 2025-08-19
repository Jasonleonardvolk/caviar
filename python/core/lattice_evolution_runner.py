"""
Lattice Evolution Runner
Manages the oscillator lattice evolution in an async context
"""

import asyncio
import logging
from typing import Optional

# Import the oscillator lattice
try:
    from .oscillator_lattice import get_global_lattice, initialize_global_lattice
    LATTICE_AVAILABLE = True
except ImportError:
    LATTICE_AVAILABLE = False
    get_global_lattice = None
    initialize_global_lattice = None

logger = logging.getLogger(__name__)


async def run_forever():
    """
    Run the oscillator lattice evolution forever
    This is the main entry point for the async lattice runner
    """
    if not LATTICE_AVAILABLE:
        logger.error("Oscillator lattice module not available")
        return
    
    logger.info("🌊 Starting oscillator lattice evolution runner...")
    
    try:
        # Initialize the global lattice if not already done
        if not initialize_global_lattice():
            logger.error("Failed to initialize oscillator lattice")
            return
        
        # Get the global lattice instance
        lattice = get_global_lattice()
        if not lattice:
            logger.error("Failed to get global lattice instance")
            return
        
        logger.info("✅ Oscillator lattice runner started successfully")
        
        # The lattice runs its own internal loop, so we just need to keep
        # this coroutine alive while monitoring the lattice status
        while True:
            try:
                # Check lattice status
                state = lattice.get_state()
                if not state.get('running', False):
                    logger.warning("Oscillator lattice stopped running, restarting...")
                    lattice.start()
                
                # Log status periodically (every 60 seconds)
                await asyncio.sleep(60)
                logger.debug(f"Oscillator lattice status: running={state.get('running')}, "
                           f"synchronization={state.get('synchronization', 0):.3f}")
                
            except Exception as e:
                logger.error(f"Error in lattice monitoring: {e}")
                await asyncio.sleep(5)  # Wait before retrying
                
    except Exception as e:
        logger.error(f"Fatal error in lattice runner: {e}")
        raise


async def run_with_timeout(timeout: Optional[float] = None):
    """
    Run the oscillator lattice with an optional timeout
    Useful for testing or limited-duration runs
    """
    if timeout:
        try:
            await asyncio.wait_for(run_forever(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.info(f"Lattice runner stopped after {timeout} seconds")
    else:
        await run_forever()


def check_lattice_availability() -> bool:
    """Check if the oscillator lattice is available and can be initialized"""
    if not LATTICE_AVAILABLE:
        return False
    
    try:
        # Try to get or create the global lattice
        lattice = get_global_lattice()
        return lattice is not None
    except Exception as e:
        logger.error(f"Error checking lattice availability: {e}")
        return False


# Export main entry point
__all__ = ['run_forever', 'run_with_timeout', 'check_lattice_availability']
