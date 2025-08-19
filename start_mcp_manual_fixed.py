#!/usr/bin/env python3
"""
Simple MCP Server Runner - Fixed Version
Run this to start MCP on port 8100
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main entry point"""
    print("=== Starting MCP Service on Port 8100 ===\n")
    
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
    print(f"PYTHONPATH: {os.environ['PYTHONPATH']}")
    
    # Try different server files in order of preference
    server_files = [
        'mcp_metacognitive/server_integrated.py',  # Best - FastMCP + FastAPI endpoints
        'mcp_metacognitive/server_fixed.py',       # Real FastMCP
        'mcp_metacognitive/server.py'              # Original server
    ]
    
    server_to_run = None
    for server_file in server_files:
        server_path = script_dir / server_file
        if server_path.exists():
            print(f"Found server file: {server_file}")
            server_to_run = server_path
            break
    
    if not server_to_run:
        print("ERROR: No MCP server file found!")
        print("Checked for:", server_files)
        return 1
    
    # Run the server directly
    cmd = [sys.executable, str(server_to_run)]
    print(f"\nRunning: {' '.join(cmd)}")
    print("\nMCP server starting...")
    print("Press Ctrl+C to stop\n")
    
    try:
        # Run server and let it handle output
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\nShutting down MCP server...")
        return 0
    except Exception as e:
        print(f"\nERROR: Failed to start MCP server: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
