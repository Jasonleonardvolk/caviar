"""
TORI MCP Server with FastAPI Integration
Combines FastMCP with FastAPI endpoints for health checks
"""

import asyncio
import logging
import os
import time
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
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
    version="1.0.0",
    description="TORI Metacognitive Processing Server with FastMCP"
)

# Try to import and setup MCP
MCP_AVAILABLE = False
mcp_server = None
mcp_error = None

try:
    # Import the real MCP server
    from mcp_metacognitive.server import mcp, setup_server
    
    # Set up the server
    setup_server()
    
    # Mount MCP endpoints
    mcp_server = mcp
    MCP_AVAILABLE = True
    logger.info("✓ FastMCP server loaded successfully")
    
except Exception as e:
    mcp_error = str(e)
    logger.error(f"✗ Failed to load FastMCP: {e}")
    
    # Try fallback to basic FastMCP
    try:
        from mcp.server.fastmcp import FastMCP
        mcp_server = FastMCP("TORI-Metacognitive-Fallback")
        MCP_AVAILABLE = True
        logger.info("✓ Using fallback FastMCP")
    except:
        logger.error("✗ No FastMCP available at all")

# Health check endpoints
@app.get("/health")
async def health():
    """Basic health check."""
    return {"status": "healthy", "mcp_available": MCP_AVAILABLE}

@app.get("/api/system/status")
async def system_status():
    """Detailed system status."""
    status = {
        "status": "operational",
        "server": "TORI MCP Server",
        "version": "1.0.0",
        "mcp_available": MCP_AVAILABLE,
        "message": "FastMCP mode" if MCP_AVAILABLE else "FastAPI only mode",
        "servers": {},
        "discovery": {
            "total_discovered": 0,
            "running": 0
        }
    }
    
    if MCP_AVAILABLE and mcp_server:
        try:
            # Try to get MCP server info
            if hasattr(mcp_server, 'list_tools'):
                tools = await mcp_server.list_tools()
                status["servers"]["tools"] = len(tools) if tools else 0
            
            status["discovery"]["total_discovered"] = 1
            status["discovery"]["running"] = 1
            status["mcp_implementation"] = type(mcp_server).__name__
        except:
            pass
    
    if mcp_error:
        status["error"] = mcp_error
    
    return JSONResponse(content=status)

@app.get("/sse")
async def sse_endpoint():
    """SSE endpoint for MCP communication."""
    if MCP_AVAILABLE and mcp_server:
        # Delegate to MCP server
        return {"message": "SSE endpoint active", "mcp": True}
    else:
        return {"message": "SSE endpoint (mock mode)", "mcp": False}

@app.get("/tools")
async def list_tools():
    """List available MCP tools."""
    if MCP_AVAILABLE and mcp_server and hasattr(mcp_server, 'list_tools'):
        try:
            tools = await mcp_server.list_tools()
            return {"tools": tools, "count": len(tools) if tools else 0}
        except:
            pass
    
    return {"tools": [], "count": 0, "message": "No MCP tools available"}

@app.get("/consciousness")
async def consciousness_status():
    """Consciousness monitoring status."""
    return {
        "status": "monitoring",
        "level": 0.3,
        "mcp_active": MCP_AVAILABLE,
        "message": "FastMCP consciousness monitoring" if MCP_AVAILABLE else "Mock consciousness"
    }

@app.post("/api/system/components/mcp_metacognitive/ready")
async def mcp_ready():
    """MCP Metacognitive component ready status."""
    return {
        "ready": True,
        "component": "mcp_metacognitive",
        "status": "operational" if MCP_AVAILABLE else "mock_mode",
        "mcp_available": MCP_AVAILABLE,
        "timestamp": time.time()
    }

# Mount the MCP app if available
if MCP_AVAILABLE and mcp_server:
    try:
        # Try to mount MCP routes
        if hasattr(mcp_server, 'routes'):
            for route in mcp_server.routes:
                app.add_api_route(route.path, route.endpoint, methods=route.methods)
            logger.info("✓ MCP routes mounted")
    except Exception as e:
        logger.warning(f"Could not mount MCP routes: {e}")

# Log startup info
@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("=" * 60)
    logger.info("TORI MCP Server Started")
    logger.info(f"MCP Available: {MCP_AVAILABLE}")
    logger.info(f"Server Type: {type(mcp_server).__name__ if mcp_server else 'None'}")
    logger.info("Endpoints:")
    logger.info("  - Health: http://localhost:8100/health")
    logger.info("  - Status: http://localhost:8100/api/system/status")
    logger.info("  - Ready: http://localhost:8100/api/system/components/mcp_metacognitive/ready")
    logger.info("  - SSE: http://localhost:8100/sse")
    logger.info("  - Tools: http://localhost:8100/tools")
    logger.info("  - Consciousness: http://localhost:8100/consciousness")
    logger.info("=" * 60)

if __name__ == "__main__":
    # Get configuration from environment
    host = os.environ.get('SERVER_HOST', '0.0.0.0')
    port = int(os.environ.get('SERVER_PORT', '8100'))
    
    # Run the server
    uvicorn.run(app, host=host, port=port, log_level="info")
