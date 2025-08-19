"""
TORI Metacognitive MCP Server - Simple Mode
==========================================

Simple FastAPI server for MCP functionality.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="TORI MCP Server",
    version="0.1.0",
    description="TORI Metacognitive Processing Server"
)

# Add required endpoints
@app.get("/api/system/status")
async def system_status():
    """Get system status."""
    return {
        "status": "operational",
        "server": "TORI MCP Server",
        "version": "0.1.0",
        "mcp_available": False,
        "message": "Using simplified FastAPI server",
        "servers": {},
        "discovery": {
            "total_discovered": 0,
            "running": 0
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/sse")
async def sse_endpoint():
    """SSE endpoint for cognitive engine events."""
    try:
        from sse_starlette.sse import EventSourceResponse
    except ImportError:
        logger.warning("sse_starlette not available, returning mock response")
        return {"message": "SSE not available in simple mode"}
    
    async def event_generator():
        """Generate SSE events."""
        while True:
            yield {
                "event": "heartbeat",
                "data": "alive"
            }
            await asyncio.sleep(30)
    
    return EventSourceResponse(event_generator())

@app.get("/tools")
async def list_tools():
    """List available tools."""
    return {
        "tools": [],
        "message": "No tools in simple mode"
    }

@app.get("/consciousness")
async def consciousness_status():
    """Get consciousness monitoring status."""
    return {
        "status": "monitoring",
        "level": 0.0,
        "message": "Simple mode - no real consciousness monitoring"
    }

def main():
    """Main entry point for the server."""
    # Get configuration from environment
    host = os.environ.get('SERVER_HOST', '0.0.0.0')
    port = int(os.environ.get('SERVER_PORT', '8100'))
    
    logger.info(f"Starting TORI MCP Server on {host}:{port}")
    logger.info("MCP should now be running on:")
    logger.info(f"  - SSE endpoint: http://localhost:{port}/sse")
    logger.info(f"  - Status: http://localhost:{port}/api/system/status")
    logger.info(f"  - Tools: http://localhost:{port}/tools")
    
    # Run the server
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
