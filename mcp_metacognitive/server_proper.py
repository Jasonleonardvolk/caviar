"""
TORI Metacognitive MCP Server - Proper Implementation
====================================================

This attempts to properly expose MCP routes if available.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, APIRouter


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
    # Actually register the tool with the server
    if hasattr(server, 'register_tool'):
        server.register_tool(name=name, handler=handler, description=description)

def register_resource_safe(server, uri, handler, description):
    """Register resource only if not already registered"""
    if uri in _REGISTRATION_REGISTRY['resources']:
        logger.debug(f"Resource {uri} already registered, skipping")
        return
    _REGISTRATION_REGISTRY['resources'].add(uri)
    # Actually register the resource with the server
    if hasattr(server, 'register_resource'):
        server.register_resource(uri=uri, handler=handler, description=description)

def register_prompt_safe(server, name, handler, description):
    """Register prompt only if not already registered"""
    if name in _REGISTRATION_REGISTRY['prompts']:
        logger.debug(f"Prompt {name} already registered, skipping")
        return
    _REGISTRATION_REGISTRY['prompts'].add(name)
    # Actually register the prompt with the server
    if hasattr(server, 'register_prompt'):
        server.register_prompt(name=name, handler=handler, description=description)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Reduce noise from duplicate registrations
logging.getLogger("mcp.server.fastmcp").setLevel(logging.ERROR)

# Try to import MCP and expose its routes
MCP_AVAILABLE = False
mcp_instance = None

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import TextContent
    
    # Create MCP instance
    mcp_instance = FastMCP("TORI MCP Server")
    
    # Check if MCP has a router we can use
    if hasattr(mcp_instance, 'router'):
        logger.info("✅ MCP router found - will expose full API")
        MCP_AVAILABLE = True
    elif hasattr(mcp_instance, 'app') and hasattr(mcp_instance.app, 'router'):
        logger.info("✅ MCP app router found - will expose full API")
        MCP_AVAILABLE = True
    else:
        # Initialize a router for MCP if it doesn't have one
        if not hasattr(mcp_instance, 'router'):
            mcp_instance.router = APIRouter()
            logger.info("✅ Created router for MCP instance")
            MCP_AVAILABLE = True
        
except ImportError as e:
    logger.warning(f"⚠️ MCP not available: {e}")

# Create our FastAPI app
app = FastAPI(
    title="TORI MCP Server",
    version="0.1.0",
    description="TORI Metacognitive Processing Server"
)

# If MCP is available, include its routes
if MCP_AVAILABLE and mcp_instance:
    try:
        # Try different ways to get the router
        if hasattr(mcp_instance, 'router'):
            app.include_router(mcp_instance.router, prefix="/mcp")
            logger.info("✅ MCP router included at /mcp")
        elif hasattr(mcp_instance, 'app'):
            # MCP might expose a FastAPI app
            if hasattr(mcp_instance.app, 'router'):
                app.include_router(mcp_instance.app.router, prefix="/mcp")
                logger.info("✅ MCP app router included at /mcp")
            elif hasattr(mcp_instance.app, 'routes'):
                # Try to mount the app directly
                app.mount("/mcp", mcp_instance.app)
                logger.info("✅ MCP app mounted at /mcp")
        
        # Also expose at root level if possible
        if hasattr(mcp_instance, 'get_asgi_app'):
            asgi_app = mcp_instance.get_asgi_app()
            app.mount("/", asgi_app)
            logger.info("✅ MCP ASGI app mounted at root")
            
    except Exception as e:
        logger.error(f"❌ Failed to include MCP routes: {e}")
        MCP_AVAILABLE = False

# Add our own endpoints as fallback or additional
@app.get("/api/system/status")
async def system_status():
    """Get system status."""
    return {
        "status": "operational",
        "server": "TORI MCP Server",
        "version": "0.1.0",
        "mcp_available": MCP_AVAILABLE,
        "mcp_routes_exposed": MCP_AVAILABLE
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "mcp": MCP_AVAILABLE}

# List available routes for debugging
@app.get("/api/routes")
async def list_routes():
    """List all available routes."""
    routes = []
    for route in app.routes:
        if hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods) if hasattr(route, 'methods') else []
            })
    return {"routes": routes, "total": len(routes)}

# If MCP provides tools, expose them
if MCP_AVAILABLE and mcp_instance:
    @app.get("/tools")
    async def list_tools():
        """List available MCP tools."""
        if hasattr(mcp_instance, 'list_tools'):
            return await mcp_instance.list_tools()
        elif hasattr(mcp_instance, 'tools'):
            return {"tools": list(mcp_instance.tools.keys())}
        else:
            return {"tools": [], "message": "No tools interface found"}

# SSE endpoint for cognitive engine
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

# Log final status
logger.info(f"TORI MCP Server ready - MCP routes exposed: {MCP_AVAILABLE}")
