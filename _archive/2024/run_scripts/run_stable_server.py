#!/usr/bin/env python3
"""
Stable FastAPI server launcher - avoids uvicorn file watcher issues
"""

import os
import sys
import uvicorn
from pathlib import Path
import subprocess
import time
import requests
import asyncio
import atexit
from datetime import datetime
import logging
from typing import Optional
from mcp_bridge_real_tori import create_real_mcp_bridge, RealMCPBridge

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global bridge instance
mcp_bridge: Optional[RealMCPBridge] = None

def start_mcp_services():
    """Start MCP TypeScript services"""
    logger.info("Starting MCP services...")
    
    # Start MCP in background
    mcp_process = subprocess.Popen(
        ['npm', 'run', 'start'],
        cwd='mcp-server-architecture',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for MCP to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get('http://localhost:3000/health')
            if response.status_code == 200:
                logger.info("MCP services started successfully")
                return mcp_process
        except:
            pass
        time.sleep(1)
    
    raise Exception("MCP services failed to start")

async def initialize_mcp_bridge():
    """Initialize MCP bridge with TORI filtering"""
    global mcp_bridge
    
    config = {
        'mcp_gateway_url': os.getenv('MCP_GATEWAY_URL', 'http://localhost:3001'),
        'auth_token': os.getenv('MCP_AUTH_TOKEN', 'your-secure-token'),
        'enable_audit_log': True
    }
    
    mcp_bridge = await create_real_mcp_bridge(config)
    
    # Register your callback handlers
    mcp_bridge.register_callback_handler(
        'kaizen.improvement',
        handle_kaizen_improvement_callback
    )
    mcp_bridge.register_callback_handler(
        'celery.task_update',
        handle_celery_task_callback
    )
    
    return mcp_bridge

async def handle_kaizen_improvement_callback(data):
    """Handle Kaizen improvement callbacks"""
    logger.info(f"Received Kaizen improvement: {data}")
    # Add your callback handling logic here
    return {"status": "processed", "data": data}

async def handle_celery_task_callback(data):
    """Handle Celery task callbacks"""
    logger.info(f"Received Celery task update: {data}")
    # Add your callback handling logic here
    return {"status": "processed", "data": data}

async def enhance_with_mcp(content: str, operation: str) -> str:
    """
    Use MCP services with TORI filtering
    This is what you'll call from your existing routes
    """
    if not mcp_bridge:
        raise Exception("MCP bridge not initialized")
    
    result = await mcp_bridge.process_to_mcp(
        content=content,
        operation=operation,
        metadata={
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'api'
        }
    )
    
    return result.filtered

def main():
    """Launch the PDF ingestion server with MCP integration"""
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent.absolute()
    
    # Change to the correct directory
    os.chdir(script_dir)
    
    print("🚀 Starting TORI FastAPI Server with MCP Integration")
    print(f"📂 Working directory: {script_dir}")
    print(f"🌐 Server will be available at: http://localhost:8002")
    print("🔧 File watching disabled for stability")
    print("=" * 50)
    
    # Start MCP services first
    try:
        mcp_process = start_mcp_services()
        print("✅ MCP services started successfully")
    except Exception as e:
        print(f"❌ Failed to start MCP services: {e}")
        return 1
    
    # Initialize bridge
    try:
        asyncio.run(initialize_mcp_bridge())
        print("✅ MCP bridge initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize MCP bridge: {e}")
        if mcp_process:
            mcp_process.terminate()
        return 1
    
    # Add shutdown handler
    def shutdown_handler():
        if mcp_bridge:
            asyncio.run(mcp_bridge.stop())
        if mcp_process:
            mcp_process.terminate()
    
    atexit.register(shutdown_handler)
    
    try:
        # Run without reload to avoid file watching issues
        uvicorn.run(
            "ingest_pdf.main:app",
            host="0.0.0.0",
            port=8002,
            reload=False,  # ✅ NO FILE WATCHING - STABLE
            workers=1,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server shutdown requested")
    except Exception as e:
        print(f"❌ Server error: {e}")
        return 1
    finally:
        # Cleanup
        shutdown_handler()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
