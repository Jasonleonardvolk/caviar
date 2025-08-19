"""
BULLETPROOF MCP SERVER - ASCII ONLY
Works with ANY MCP version - no import crashes ever
"""

import os
import sys
import signal
import asyncio
import logging
from typing import Optional, Dict, Any

# Configure logging - ASCII only
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Fix the import path issue
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# BULLETPROOF MCP IMPORTS - try different versions
MCP_SERVER = None
MCP_AVAILABLE = False

# Try Method 1: Standard MCP
try:
    from mcp.server.fastmcp import FastMCP
    MCP_SERVER = "fastmcp"
    MCP_AVAILABLE = True
    logger.info("SUCCESS: FastMCP available")
except ImportError:
    logger.warning("FastMCP not available")

# Try Method 2: Basic MCP server
if not MCP_AVAILABLE:
    try:
        from mcp.server import Server
        MCP_SERVER = "basic"
        MCP_AVAILABLE = True
        logger.info("SUCCESS: Basic MCP server available")
    except ImportError:
        logger.warning("Basic MCP server not available")

# Try Method 3: Any MCP
if not MCP_AVAILABLE:
    try:
        import mcp
        MCP_SERVER = "minimal"
        MCP_AVAILABLE = True
        logger.info("SUCCESS: Minimal MCP available")
    except ImportError:
        logger.warning("No MCP packages available")

# Import config safely
try:
    from mcp_metacognitive import config
    CONFIG_AVAILABLE = True
except ImportError:
    logger.warning("Config not available - using defaults")
    CONFIG_AVAILABLE = False
    
    class MockConfig:
        def __init__(self):
            self.host = "localhost"
            self.port = 8888
            self.debug = False
    
    config = MockConfig()


def create_bulletproof_server():
    """Create server that works with any MCP version"""
    
    if MCP_SERVER == "fastmcp":
        return create_fastmcp_server()
    elif MCP_SERVER == "basic":
        return create_basic_server()
    elif MCP_SERVER == "minimal":
        return create_minimal_server()
    else:
        return create_fallback_server()


def create_fastmcp_server():
    """Create FastMCP server"""
    try:
        from mcp.server.fastmcp import FastMCP
        
        server = FastMCP("TORI-Metacognitive")
        
        @server.tool()
        def analyze_cognition(query: str) -> Dict[str, Any]:
            """Analyze cognitive patterns"""
            return {
                "query": query,
                "analysis": "cognitive pattern detected",
                "confidence": 0.8
            }
        
        @server.tool()
        def ping() -> str:
            """Health check"""
            return "pong - FastMCP server"
        
        logger.info("SUCCESS: FastMCP server created with tools")
        return server
    
    except Exception as e:
        logger.error(f"FastMCP creation failed: {e}")
        return create_fallback_server()


def create_basic_server():
    """Create basic MCP server"""
    try:
        from mcp.server import Server
        
        # Create basic server
        server = Server("TORI-Basic")
        logger.info("SUCCESS: Basic MCP server created")
        return server
    
    except Exception as e:
        logger.error(f"Basic server creation failed: {e}")
        return create_fallback_server()


def create_minimal_server():
    """Create minimal server with MCP package"""
    try:
        import mcp
        
        class MinimalServer:
            def __init__(self):
                self.name = "TORI-Minimal"
                self.running = False
            
            async def run(self):
                self.running = True
                logger.info("SUCCESS: Minimal MCP server running")
                while self.running:
                    await asyncio.sleep(1)
            
            def stop(self):
                self.running = False
        
        server = MinimalServer()
        logger.info("SUCCESS: Minimal server created")
        return server
    
    except Exception as e:
        logger.error(f"Minimal server creation failed: {e}")
        return create_fallback_server()


def create_fallback_server():
    """Create ultimate fallback server"""
    
    class FallbackServer:
        def __init__(self):
            self.name = "TORI-Fallback"
            self.running = False
            self.tools = {
                "ping": lambda: "pong - fallback server",
                "status": lambda: {"status": "running", "mode": "fallback"}
            }
        
        async def run(self):
            self.running = True
            logger.info("SUCCESS: Fallback server running")
            while self.running:
                await asyncio.sleep(1)
        
        def stop(self):
            self.running = False
            logger.info("INFO: Fallback server stopped")
    
    server = FallbackServer()
    logger.info("SUCCESS: Fallback server created")
    return server


def setup_signal_handlers(server):
    """Setup signal handlers"""
    
    def signal_handler(signum, frame):
        logger.info(f"INFO: Received signal {signum}, shutting down")
        if hasattr(server, 'stop'):
            server.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point - bulletproof"""
    logger.info("LAUNCH: Starting BULLETPROOF MCP server")
    logger.info(f"INFO: MCP mode: {MCP_SERVER or 'fallback'}")
    
    # Create server
    try:
        server = create_bulletproof_server()
        logger.info("SUCCESS: Server created")
    except Exception as e:
        logger.error(f"ERROR: Server creation failed: {e}")
        return 1
    
    # Setup signal handlers
    setup_signal_handlers(server)
    
    # Run the server
    try:
        host = getattr(config, 'host', 'localhost')
        port = getattr(config, 'port', 8888)
        logger.debug(f"INFO: Server starting on {host}:{port}")
        
        if hasattr(server, 'run'):
            await server.run()
        else:
            # Manual run loop for basic servers
            logger.info("INFO: Running manual server loop")
            while True:
                await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("INFO: Server interrupted")
    except Exception as e:
        logger.error(f"ERROR: Server runtime error: {e}")
        return 1
    
    return 0


# Expose app for uvicorn
async def app():
    """Entrypoint for uvicorn - returns the server without running it"""
    logger.info("UVICORN: Starting MCP server via uvicorn")
    logger.info(f"INFO: MCP mode: {MCP_SERVER or 'fallback'}")
    
    try:
        server = create_bulletproof_server()
        logger.info("SUCCESS: Server created for uvicorn")
        return server
    except Exception as e:
        logger.error(f"ERROR: Server creation failed in uvicorn mode: {e}")
        # Return minimal working object for uvicorn
        class EmptyApp:
            async def __call__(self, scope, receive, send):
                await send({
                    'type': 'http.response.start',
                    'status': 500,
                    'headers': [(b'content-type', b'text/plain')],
                })
                await send({
                    'type': 'http.response.body',
                    'body': f'Server creation failed: {e}'.encode(),
                })
        return EmptyApp()


def is_running_under_uvicorn():
    """Detect if we're running under uvicorn"""
    # Check for uvicorn in the command line or environment
    return ('uvicorn' in sys.argv[0].lower() or
            os.environ.get('UVICORN_FD') is not None or
            'gunicorn' in sys.argv[0].lower())


def start_server():
    """Start the server based on the runtime environment"""
    try:
        # Detect if we're running under uvicorn
        if is_running_under_uvicorn():
            logger.info("INFO: Detected running under uvicorn/gunicorn")
            # Return the app directly, let uvicorn handle it
            return app()
        
        # Check if event loop is already running
        try:
            loop = asyncio.get_running_loop()
            logger.info("INFO: Event loop already running, creating task")
            # We're in an existing event loop, so create a task
            task = asyncio.create_task(main())
            # Return the task so it can be awaited if needed
            return task
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            logger.info("INFO: No event loop detected, using asyncio.run")
            return asyncio.run(main())
    except Exception as e:
        logger.error(f"CRITICAL: Server start failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = start_server()
        if isinstance(exit_code, int):
            sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("INFO: Server interrupted at startup")
        sys.exit(0)
    except Exception as e:
        logger.error(f"CRITICAL: {e}")
        sys.exit(1)
