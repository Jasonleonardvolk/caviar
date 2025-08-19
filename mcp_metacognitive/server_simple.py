"""
TORI Metacognitive MCP Server
============================

Simple FastAPI server for MCP functionality.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI


# Global registration tracking to prevent duplicates
_REGISTRATION_REGISTRY = {
    'tools': set(),
    'resources': set(), 
    'prompts': set()
}

def register_tool_safe(server, name, handler, description):
    """Register tool only if not already registered"""
    if name in _REGISTRATION_REGISTRY['tools']:
        logger.debug(f"Tool {name} already registered, skipping")
        return
    _REGISTRATION_REGISTRY['tools'].add(name)
    register_tool_safe(server, name=name, handler=handler, description=description)

def register_resource_safe(server, uri, handler, description):
    """Register resource only if not already registered"""
    if uri in _REGISTRATION_REGISTRY['resources']:
        logger.debug(f"Resource {uri} already registered, skipping")
        return
    _REGISTRATION_REGISTRY['resources'].add(uri)
    register_resource_safe(server, uri=uri, handler=handler, description=description)

def register_prompt_safe(server, name, handler, description):
    """Register prompt only if not already registered"""
    if name in _REGISTRATION_REGISTRY['prompts']:
        logger.debug(f"Prompt {name} already registered, skipping")
        return
    _REGISTRATION_REGISTRY['prompts'].add(name)
    register_prompt_safe(server, name=name, handler=handler, description=description)

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
        "message": "Using simplified FastAPI server"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/sse")
async def sse_endpoint():
    """SSE endpoint for cognitive engine events."""
    from sse_starlette.sse import EventSourceResponse
    import asyncio
    
    async def event_generator():
        """Generate SSE events."""
        while True:
            yield {
                "event": "heartbeat",
                "data": "alive"
            }
            await asyncio.sleep(30)
    
    return EventSourceResponse(event_generator())

# Log startup
logger.info("TORI MCP Server ready (simplified mode)")

# Try to import MCP if available (for future use)
try:
    from mcp.server.fastmcp import FastMCP
    logger.info("MCP package found but using simplified server")
except ImportError:
    logger.info("MCP package not available, using FastAPI")
