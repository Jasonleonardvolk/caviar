"""
TORI Metacognitive MCP Server
============================

Main server implementation using FastMCP.
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
    # Use our local FastMCP implementation
    from mcp_metacognitive.fastmcp import FastMCP
    TextContent = dict  # Dummy for compatibility
    MCP_AVAILABLE = True  # We have our own implementation

from mcp_metacognitive.core.config import config
from mcp_metacognitive.core.state_manager import state_manager

# Import tools, resources, and prompts
from mcp_metacognitive.tools import register_tools
from mcp_metacognitive.resources import register_resources
from mcp_metacognitive.prompts import register_prompts

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

# Create server
mcp = FastMCP(
    name=config.server_name,
    version=config.server_version
)

# The FastMCP instance is the ASGI app
app = mcp

logger.info("FastMCP server configured")

# Ensure app is set at module level for uvicorn
globals()['app'] = app

# Add flag to track if server has been set up
_server_setup_complete = False

# Add server metadata
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

def setup_server():
    """Set up the MCP server with all tools, resources, and prompts."""
    global _server_setup_complete
    
    # Check if already set up to prevent duplicate registration
    if _server_setup_complete:
        logger.debug("Server already set up, skipping duplicate registration")
        return
    
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
    
    # Mark setup as complete
    _server_setup_complete = True
    logger.info(f"TORI MCP server setup complete")

# Main entry point
def main():
    """Main entry point for the MCP server."""
    # Import here to avoid circular dependencies
    # Note: MemoryManager import removed - not used in this file
    
    # Fallback for when MCP not available
    def fallback_main():
        logger.warning("Running in fallback mode without MCP")
        # Basic event loop for testing
        async def run():
            logger.info("Fallback server running (no MCP functionality)")
            await asyncio.sleep(3600)  # Run for an hour
        
        asyncio.run(run())
    
    if not MCP_AVAILABLE:
        logger.warning("MCP packages not available, using fallback implementation")
        return fallback_main()
    
    setup_server()
    
    if config.transport_type == "stdio":
        logger.info("Starting TORI MCP server with stdio transport...")
        mcp.run()
    elif config.transport_type == "sse":
        logger.info(f"Starting TORI MCP server with SSE transport on {config.server_host}:{config.server_port}...")
        # Fix: Use old signature (host/port) for installed version
        try:
            # Try with parameters first
            mcp.run(transport="sse", host=config.server_host, port=config.server_port)
        except TypeError:
            # If that fails, try without parameters and let MCP handle it
            logger.debug("FastMCP.run() doesn't accept host/port, using defaults")
            mcp.run()
    else:
        raise ValueError(f"Unsupported transport type: {config.transport_type}")

if __name__ == "__main__":
    # For SSE transport, we need to ensure we have a proper ASGI app
    if config.transport_type == "sse" and MCP_AVAILABLE:
        # Setup the server before running
        setup_server()
        
        # If we have a FastAPI app, use uvicorn directly
        if app is not None and hasattr(app, '__call__'):
            import uvicorn
            host = os.getenv("SERVER_HOST", "0.0.0.0")
            port = int(os.getenv("SERVER_PORT", "8100"))
            
            logger.info(f"Starting TORI MCP server with uvicorn on {host}:{port}...")
            try:
                uvicorn.run(app, host=host, port=port, log_level="info")
            except Exception as e:
                logger.error(f"Failed to start with uvicorn: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"App type: {type(app)}")
                # Fall back to MCP's built-in server
                main()
        else:
            # Use MCP's built-in SSE server
            logger.info("Using FastMCP's built-in SSE server")
            main()
    else:
        # Use the normal main() for stdio or fallback
        main()