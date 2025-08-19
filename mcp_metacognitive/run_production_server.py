#!/usr/bin/env python3
"""
🏆 TORI MCP Production Server Launcher
=====================================

Stable production launcher with REAL TORI filtering and Soliton Memory integration.
Based on run_stable_server.py patterns with enhanced MCP integration.
"""

import os
import sys
import asyncio
import uvicorn
import signal
import time
import logging
import atexit
from pathlib import Path
from datetime import datetime
from typing import Optional

from core.config import config
from core.state_manager import state_manager
from core.real_tori_bridge import RealTORIFilter

# Setup enhanced logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.log_file) if config.log_file else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global references for cleanup
mcp_server_process = None
shutdown_requested = False

async def initialize_production_systems():
    """Initialize all production systems with health checks"""
    logger.info("🏆 Initializing TORI MCP Production Systems...")
    
    # 1. Initialize State Manager (already done via import)
    logger.info(f"✅ Cognitive State Manager: {state_manager.dimension}D")
    logger.info(f"✅ REAL TORI Filtering: {state_manager.tori_filter.real_tori_available}")
    logger.info(f"✅ Soliton Memory System: Infinite Context")
    
    # 2. Pre-populate with initial memories if needed
    try:
        initial_memory_id = await state_manager.store_memory(
            concept_id="system_initialization",
            content=f"TORI MCP Production Server started at {datetime.now().isoformat()}",
            importance=0.8,
            content_type=state_manager.soliton_lattice.ContentType.TEXT
        )
        logger.info(f"✅ Initial system memory stored: {initial_memory_id}")
    except Exception as e:
        logger.warning(f"Failed to store initial memory: {e}")
    
    # 3. Test core functionality
    try:
        # Test cognitive state
        current_state = await state_manager.get_current_state()
        logger.info(f"✅ Current Φ: {current_state['phi']:.3f}")
        
        # Test memory recall
        memories = await state_manager.recall_memories("system_initialization", max_results=1)
        logger.info(f"✅ Memory recall: {len(memories)} memories found")
        
        # Test TORI filtering
        test_content = "Hello, TORI production system!"
        filtered_content = await state_manager.filter_content(test_content, "input")
        logger.info(f"✅ TORI filtering: Active")
        
    except Exception as e:
        logger.error(f"❌ System health check failed: {e}")
        raise
    
    logger.info("🏆 All production systems initialized successfully!")

def setup_signal_handlers():
    """Setup graceful shutdown handlers"""
    def signal_handler(signum, frame):
        global shutdown_requested
        logger.info(f"🏆 Received signal {signum}, initiating graceful shutdown...")
        shutdown_requested = True
        
        # Trigger async shutdown
        asyncio.create_task(graceful_shutdown())
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

async def graceful_shutdown():
    """Gracefully shutdown all systems"""
    global mcp_server_process, shutdown_requested
    
    logger.info("🏆 Starting graceful shutdown...")
    
    try:
        # 1. Save current state and memory
        await state_manager.save_state("production_shutdown.json")
        logger.info("✅ State and memory saved")
        
        # 2. Log final statistics
        stats = await state_manager.get_memory_statistics()
        logger.info(f"📊 Final Statistics:")
        logger.info(f"   - Total Memories: {stats.get('memory_system', {}).get('total_memories', 0)}")
        logger.info(f"   - Session Duration: {stats.get('cognitive_system', {}).get('session_duration', 0):.1f}s")
        logger.info(f"   - State Updates: {stats.get('cognitive_system', {}).get('state_updates', 0)}")
        
        # 3. Cleanup processes
        if mcp_server_process:
            mcp_server_process.terminate()
            logger.info("✅ MCP server process terminated")
        
        logger.info("🏆 Graceful shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")
    
    finally:
        shutdown_requested = True

def health_monitor():
    """Continuous health monitoring"""
    async def monitor():
        while not shutdown_requested:
            try:
                # Check system health
                current_state = await state_manager.get_current_state()
                
                # Check consciousness level
                if current_state['phi'] < config.consciousness_threshold * 0.5:
                    logger.warning(f"⚠️ Low consciousness detected: Φ = {current_state['phi']:.3f}")
                
                # Check memory system
                stats = await state_manager.get_memory_statistics()
                memory_count = stats.get('memory_system', {}).get('total_memories', 0)
                
                if memory_count > 0:
                    logger.debug(f"💡 Health Check: Φ={current_state['phi']:.3f}, Memories={memory_count}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Health monitor error: {e}")
                await asyncio.sleep(10)  # Shorter retry interval on error
    
    return asyncio.create_task(monitor())

def create_production_app():
    """Create production FastAPI app with all enhancements"""
    # Import the main server after initialization
    from server import mcp
    
    # Add production middleware and enhancements here if needed
    logger.info("🏆 Production MCP app created")
    return mcp.app

def main():
    """Main production launcher"""
    global mcp_server_process
    
    print("🚀 TORI MCP Production Server")
    print("=" * 50)
    print(f"📂 Working directory: {Path.cwd()}")
    print(f"🌐 Server will be available via MCP protocol")
    print(f"🏆 REAL TORI filtering: Enabled")
    print(f"🌊 Soliton memory: Infinite context")
    print(f"🔧 Transport: {config.transport_type}")
    print("=" * 50)
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Add cleanup handler
    atexit.register(lambda: asyncio.run(graceful_shutdown()) if not shutdown_requested else None)
    
    try:
        # Initialize async systems
        asyncio.run(initialize_production_systems())
        
        # Start health monitoring
        health_task = health_monitor()
        
        print("✅ All systems initialized successfully")
        print(f"🏆 Starting TORI MCP Production Server...")
        print(f"💡 Use Ctrl+C for graceful shutdown")
        print("=" * 50)
        
        # Import and setup the main server
        from server import main as server_main
        
        # Run the main server
        server_main()
        
    except KeyboardInterrupt:
        print("\n👋 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        logger.error(f"Critical server error: {e}")
        return 1
    finally:
        # Ensure cleanup
        if not shutdown_requested:
            try:
                asyncio.run(graceful_shutdown())
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    print("👋 TORI MCP Production Server shutdown complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())