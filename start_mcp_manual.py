#!/usr/bin/env python3
"""
Simple MCP Server Runner - Handles asyncio conflicts
Run this to start MCP on port 8100 without asyncio conflicts
"""

import os
import sys
import threading
import time
from pathlib import Path

# Add mcp_metacognitive to path
script_dir = Path(__file__).parent
mcp_path = script_dir / "mcp_metacognitive"
if mcp_path.exists():
    sys.path.insert(0, str(mcp_path))

def run_mcp_in_thread():
    """Run MCP server in a separate thread with its own event loop"""
    import asyncio
    
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Import and run the server
        from server_simple import main as mcp_main
        print("Starting MCP server in thread with new event loop...")
        loop.run_until_complete(mcp_main())
    except ImportError:
        print("Error: Could not import server_simple")
        print("Make sure mcp_metacognitive directory exists and has server_simple.py")
    except Exception as e:
        print(f"Error running MCP server: {e}")
    finally:
        loop.close()

def main():
    """Main entry point"""
    print("=== Starting MCP Service on Port 8100 ===\n")
    
    # Set environment
    os.environ['TRANSPORT_TYPE'] = 'sse'
    os.environ['SERVER_PORT'] = '8100'
    os.environ['SERVER_HOST'] = '0.0.0.0'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['TORI_INTEGRATION'] = 'true'
    
    # Start MCP in a separate thread
    mcp_thread = threading.Thread(target=run_mcp_in_thread, daemon=False)
    mcp_thread.start()
    
    print("MCP server thread started!")
    print("Waiting for server to initialize...")
    time.sleep(3)
    
    print("\nMCP should now be running on:")
    print("  - SSE endpoint: http://localhost:8100/sse")
    print("  - Status: http://localhost:8100/api/system/status")
    print("  - Tools: http://localhost:8100/tools")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down MCP server...")
        sys.exit(0)

if __name__ == "__main__":
    main()
