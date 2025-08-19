#!/usr/bin/env python3
"""
Start MCP with Full FastMCP Backend
This launches the real MCP server, not the simplified version
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main entry point"""
    print("=== Starting FULL MCP Service with FastMCP ===\n")
    
    # Set working directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Set environment
    os.environ['TRANSPORT_TYPE'] = 'sse'
    os.environ['SERVER_PORT'] = '8100'
    os.environ['SERVER_HOST'] = '0.0.0.0'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['TORI_INTEGRATION'] = 'true'
    os.environ['PYTHONPATH'] = str(script_dir)
    
    print(f"Working directory: {script_dir}")
    print("Using FULL MCP server with FastMCP backend\n")
    
    # Check if FastMCP is available
    try:
        import mcp.server.fastmcp
        print("✓ FastMCP package found")
    except ImportError:
        print("✗ FastMCP package not found")
        print("  Installing: pip install mcp fastmcp")
        print("  Continuing with local FastMCP implementation...\n")
    
    # Use uvicorn to run the full server
    cmd = [
        sys.executable, "-m", "uvicorn",
        "mcp_metacognitive.server:app",
        "--host", "0.0.0.0",
        "--port", "8100",
        "--log-level", "info"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("\nStarting FULL MCP server with FastMCP...")
    print("This should show mcp_available: True")
    print("Press Ctrl+C to stop\n")
    
    try:
        # Run server
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\nShutting down MCP server...")
        return 0
    except Exception as e:
        print(f"\nERROR: Failed to start MCP server: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the venv: .venv\\Scripts\\activate")
        print("2. Install deps: pip install fastapi uvicorn mcp fastmcp")
        print("3. Check that mcp_metacognitive/__init__.py exists")
        return 1

if __name__ == "__main__":
    sys.exit(main())
