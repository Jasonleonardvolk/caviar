"""
TORI Metacognitive MCP Server - Fallback Implementation
======================================================

This is a fallback server that works without external MCP dependencies.
It provides the same interface but uses FastAPI instead of FastMCP.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
from datetime import datetime
import os
import sys

# Try to import FastMCP, fall back to FastAPI if not available
try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: MCP packages not available, using FastAPI fallback")

# Import FastAPI for fallback
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: FastAPI not available")

# Import core components
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

# Create server instance based on available packages
if MCP_AVAILABLE:
    # Use FastMCP if available
    mcp = FastMCP(
        name=config.server_name,
        version=config.server_version
    )
    
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
            "implementation": "fastmcp",
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
    
elif FASTAPI_AVAILABLE:
    # Use FastAPI as fallback
    app = FastAPI(
        title=config.server_name,
        version=config.server_version,
        description=config.server_description
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        return {"message": "TORI MCP Metacognitive Server (FastAPI mode)"}
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "implementation": "fastapi"}
    
    @app.get("/server_info")
    async def get_server_info():
        """Get information about the TORI MCP server."""
        return {
            "name": config.server_name,
            "version": config.server_version,
            "description": config.server_description,
            "cognitive_dimension": config.cognitive_dimension,
            "manifold_metric": config.manifold_metric,
            "transport": config.transport_type,
            "implementation": "fastapi",
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
    
    # Import and register integration endpoints
    from integration.server_integration import register_integration_endpoints, tori_integration
    register_integration_endpoints(app)
    
    # SSE endpoint for MCP compatibility
    @app.get("/sse")
    async def sse_endpoint():
        """Server-Sent Events endpoint for MCP compatibility"""
        async def event_generator():
            # Initialize integration on first connection
            if not tori_integration.is_initialized:
                await tori_integration.initialize()
            
            while True:
                # Send heartbeat with system status
                status = await tori_integration.get_system_status()
                yield f"data: {json.dumps({'type': 'heartbeat', 'status': status, 'timestamp': datetime.utcnow().isoformat()})}\n\n"
                await asyncio.sleep(30)
        
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    
    # Consciousness monitoring endpoint
    @app.get("/consciousness")
    async def consciousness_status():
        """Get current consciousness status"""
        return {
            "phi": 0.42,  # Integrated Information
            "awareness_level": "active",
            "metacognitive_depth": 3,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Tools endpoint
    @app.get("/tools")
    async def list_tools():
        """List available cognitive tools"""
        return {
            "tools": [
                "reflection",
                "dynamics",
                "consciousness",
                "metacognitive",
                "soliton_memory"
            ]
        }
    
    @app.post("/invoke/{tool_name}")
    async def invoke_tool(tool_name: str, params: Dict[str, Any] = None):
        """Invoke a cognitive tool"""
        # This is a placeholder - actual tool invocation would go here
        return {
            "tool": tool_name,
            "result": f"Tool {tool_name} invoked",
            "params": params
        }

else:
    # No server framework available
    logger.error("Neither MCP nor FastAPI packages are available!")
    
    class MockServer:
        """Mock server for when no framework is available"""
        def __init__(self, name, version):
            self.name = name
            self.version = version
            logger.warning("Using mock server - no actual server functionality")
        
        def run(self, **kwargs):
            logger.error("Cannot run server - no server framework available")
            raise ImportError("Please install either 'mcp-server-sdk' or 'fastapi uvicorn'")
    
    mcp = MockServer(config.server_name, config.server_version)
    app = None

# Setup server and register integration
def setup_server():
    """Set up the server with all Phase 2 components"""
    logger.info("Setting up TORI server with Phase 2 components...")
    
    # Initialize Phase 2 integration
    from integration.server_integration import tori_integration
    
    # Configure based on environment
    integration_config = {
        "daniel": {
            "model_backend": os.getenv("DANIEL_MODEL_BACKEND", "mock"),
            "api_key": os.getenv("DANIEL_API_KEY", ""),
            "enable_metacognition": True,
            "enable_consciousness_tracking": True
        },
        "kaizen": {
            "auto_start": os.getenv("KAIZEN_AUTO_START", "true").lower() == "true",
            "analysis_interval": int(os.getenv("KAIZEN_ANALYSIS_INTERVAL", "3600")),
            "enable_auto_apply": False  # Safety: don't auto-apply in production
        }
    }
    
    # Initialize integration with config
    tori_integration.config = integration_config
    
    # Start initialization task
    asyncio.create_task(tori_integration.initialize())
    
    logger.info("Phase 2 components initialization started")

# Main entry point
def main():
    """Main entry point for the server."""
    if MCP_AVAILABLE:
        # Original MCP setup
        logger.info("Setting up TORI MCP server...")
        
        # Register tools with FastMCP
        register_tools(mcp, state_manager)
        logger.info("Tools registered with FastMCP")
        
        # Register resources
        register_resources(mcp, state_manager)
        logger.info("Resources registered with FastMCP")
        
        # Register prompts
        register_prompts(mcp, state_manager)
        logger.info("Prompts registered with FastMCP")
        
        # Initialize Phase 2 components
        setup_server()
        
        if config.transport_type == "stdio":
            logger.info("Starting TORI MCP server with stdio transport...")
            mcp.run()
        elif config.transport_type == "sse":
            logger.info(f"Starting TORI MCP server with SSE transport on {config.server_host}:{config.server_port}...")
            mcp.run(
                transport="sse",
                host=config.server_host,
                port=config.server_port
            )
        else:
            raise ValueError(f"Unsupported transport type: {config.transport_type}")
    
    elif FASTAPI_AVAILABLE:
        # Setup server with Phase 2 components
        setup_server()
        
        # Run FastAPI server
        logger.info(f"Starting TORI MCP server (FastAPI mode) on {config.server_host}:{config.server_port}...")
        uvicorn.run(
            app,
            host=config.server_host,
            port=config.server_port,
            log_level=config.log_level.lower()
        )
    
    else:
        logger.error("No server framework available!")
        sys.exit(1)

if __name__ == "__main__":
    main()
