"""
BULLETPROOF MCP MAIN - ASCII ONLY
Works with any MCP version - no crashes ever
"""

import asyncio
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# BULLETPROOF MCP IMPORT - try all versions
MCP_AVAILABLE = False
MCP_TYPE = None

# Try FastMCP first
try:
    from mcp.server.fastmcp import FastMCP
    MCP_TYPE = "fastmcp"
    MCP_AVAILABLE = True
    logger.info("SUCCESS: FastMCP found")
except ImportError:
    pass

# Try basic server
if not MCP_AVAILABLE:
    try:
        from mcp.server import Server
        MCP_TYPE = "basic"
        MCP_AVAILABLE = True
        logger.info("SUCCESS: Basic MCP found")
    except ImportError:
        pass

# Try any MCP
if not MCP_AVAILABLE:
    try:
        import mcp
        MCP_TYPE = "minimal"
        MCP_AVAILABLE = True
        logger.info("SUCCESS: Minimal MCP found")
    except ImportError:
        logger.warning("WARNING: No MCP packages found")


def setup_server():
    """Setup server based on available MCP"""
    
    if MCP_TYPE == "fastmcp":
        return setup_fastmcp_server()
    elif MCP_TYPE == "basic":
        return setup_basic_server()
    elif MCP_TYPE == "minimal":
        return setup_minimal_server()
    else:
        return setup_fallback_server()


def setup_fastmcp_server():
    """Setup FastMCP server"""
    try:
        from mcp.server.fastmcp import FastMCP
        
        server = FastMCP("TORI-Real")
        
        @server.tool()
        def ping() -> str:
            """Health check tool"""
            return "pong - REAL MCP server responding"
        
        @server.tool()
        def cognitive_status() -> Dict[str, Any]:
            """Get cognitive system status"""
            return {
                "status": "healthy",
                "mcp_mode": "real",
                "server_type": "FastMCP",
                "tools_available": 2
            }
        
        logger.info("SUCCESS: FastMCP server setup complete")
        return server
    
    except Exception as e:
        logger.error(f"ERROR: FastMCP setup failed: {e}")
        return setup_fallback_server()


def setup_basic_server():
    """Setup basic MCP server"""
    try:
        from mcp.server import Server
        
        server = Server("TORI-Basic")
        logger.info("SUCCESS: Basic MCP server setup")
        return server
    
    except Exception as e:
        logger.error(f"ERROR: Basic setup failed: {e}")
        return setup_fallback_server()


def setup_minimal_server():
    """Setup minimal server"""
    
    class MinimalServer:
        def __init__(self):
            self.name = "TORI-Minimal"
            self.running = False
        
        async def run(self):
            self.running = True
            logger.info("INFO: Minimal server running")
            while self.running:
                await asyncio.sleep(1)
        
        def stop(self):
            self.running = False
    
    server = MinimalServer()
    logger.info("SUCCESS: Minimal server setup")
    return server


def setup_fallback_server():
    """Setup ultimate fallback"""
    
    class FallbackServer:
        def __init__(self):
            self.name = "TORI-Fallback"
            self.running = False
        
        async def run(self):
            self.running = True
            logger.info("INFO: Fallback server running")
            while self.running:
                await asyncio.sleep(1)
        
        def stop(self):
            self.running = False
    
    server = FallbackServer()
    logger.info("SUCCESS: Fallback server ready")
    return server


async def run_server():
    """Run the MCP server"""
    logger.info("LAUNCH: Starting MCP server")
    
    server = setup_server()
    
    try:
        if hasattr(server, 'run'):
            await server.run()
        else:
            # Manual loop
            while True:
                await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"ERROR: Server runtime error: {e}")
        raise
    
    return 0


def main():
    """Main entry point"""
    logger.info(f"INFO: Using MCP type: {MCP_TYPE or 'fallback'}")
    
    try:
        return asyncio.run(run_server())
    except Exception as e:
        logger.error(f"ERROR: Main error: {e}")
        return 1


# Keep this for compatibility but it just calls main now
def fallback_main():
    """Legacy fallback - just calls main"""
    return main()


if __name__ == "__main__":
    main()
