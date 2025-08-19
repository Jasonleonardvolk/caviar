"""
TORI Metacognitive MCP Server - Direct Fix
==========================================

This version extracts the FastAPI app directly for uvicorn.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

# Try to import MCP packages, fall back if not available
try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Import fallback implementation
    from server_fallback import main as fallback_main, app as fallback_app

from core.config import config
from core.state_manager import state_manager

# Import tools, resources, and prompts
from tools import register_tools
from resources import register_resources
from prompts import register_prompts

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.log_file) if config.log_file else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create server based on available packages
if MCP_AVAILABLE:
    # Create FastMCP server
    mcp = FastMCP(
        name=config.server_name,
        version=config.server_version
    )
    
    # Try to get the internal FastAPI app
    # Common patterns: .app, ._app, .server, ._server
    app = None
    for attr in ['app', '_app', 'server', '_server', 'asgi', '_asgi']:
        if hasattr(mcp, attr):
            potential_app = getattr(mcp, attr)
            if potential_app is not None:
                app = potential_app
                logger.info(f"Found FastAPI app at mcp.{attr}")
                break
    
    if app is None:
        logger.warning("Could not find FastAPI app inside FastMCP, will use FastMCP directly")
        app = mcp
else:
    # Will use fallback implementation
    mcp = None
    app = fallback_app if 'fallback_app' in locals() else None

# Add server metadata
if MCP_AVAILABLE:
    @mcp.tool()
    async def get_server_info() -> str:
        """Get information about the TORI MCP server."""
        return {
            "name": config.server_name,
            "version": config.server_version,
            "description": config.server_description,
            "cognitive_dimension": config.cognitive_dimension,
            "manifold_metric": config.manifold_metric,
            "transport": config.transport_type,
            "components": [
                "manifold",
                "reflective_operator",
                "self_modification",
                "curiosity_functional",
                "cognitive_dynamics",
                "consciousness_monitor",
                "metacognitive_tower",
                "knowledge_sheaf"
            ]
        }

# Register all components
def setup_server():
    """Set up the MCP server with all tools, resources, and prompts."""
    if not MCP_AVAILABLE:
        return  # Skip setup if using fallback
    
    logger.info("Setting up TORI MCP server...")
    
    # Register tools
    register_tools(mcp, state_manager)
    logger.info("Tools registered")
    
    # Register resources
    register_resources(mcp, state_manager)
    logger.info("Resources registered")
    
    # Register prompts
    register_prompts(mcp, state_manager)
    logger.info("Prompts registered")
    
    logger.info(f"TORI MCP server setup complete")

# Main entry point
def main():
    """Main entry point for the server."""
    if not MCP_AVAILABLE:
        logger.warning("MCP packages not available, using fallback implementation")
        return fallback_main()
    
    setup_server()
    
    if config.transport_type == "stdio":
        logger.info("Starting TORI MCP server with stdio transport...")
        mcp.run()
    elif config.transport_type == "sse":
        logger.info(f"Starting TORI MCP server with SSE transport on {config.server_host}:{config.server_port}...")
        # This is where we'd normally call mcp.run() but it doesn't accept host/port
        # So we'll handle it in __main__ with uvicorn directly
        return app  # Return the app for uvicorn
    else:
        raise ValueError(f"Unsupported transport type: {config.transport_type}")

if __name__ == "__main__":
    # For SSE transport, use uvicorn directly
    if config.transport_type == "sse":
        # Setup the server
        setup_server()
        
        # Get host and port from environment
        host = os.getenv("SERVER_HOST", config.server_host)
        port = int(os.getenv("SERVER_PORT", config.server_port))
        
        # Import uvicorn
        import uvicorn
        
        # Get the app (either from FastMCP or fallback)
        if app is not None:
            logger.info(f"Starting TORI MCP server with uvicorn on {host}:{port}...")
            uvicorn.run(app, host=host, port=port, log_level="info")
        else:
            logger.error("No FastAPI app available to run with uvicorn")
            exit(1)
    else:
        # For stdio or other transports, use the normal flow
        main()
