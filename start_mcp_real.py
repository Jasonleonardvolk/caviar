#!/usr/bin/env python3
"""
Start FULL MCP Server with FastMCP Integration
This runs the real MCP with proper FastAPI endpoints
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
    env = os.environ.copy()
    env.update({
        'TRANSPORT_TYPE': 'sse',
        'SERVER_PORT': '8100',
        'SERVER_HOST': '0.0.0.0',
        'PYTHONIOENCODING': 'utf-8',
        'TORI_INTEGRATION': 'true',
        'PYTHONPATH': str(script_dir)
    })
    
    print(f"Working directory: {script_dir}")
    
    # Check dependencies
    print("\nChecking dependencies...")
    missing_deps = []
    
    deps_to_check = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("mcp.server.fastmcp", "mcp fastmcp"),
        ("sse_starlette", "sse-starlette")
    ]
    
    for module, install_name in deps_to_check:
        try:
            if "." in module:
                parts = module.split(".")
                exec(f"import {parts[0]}")
                exec(f"from {'.'.join(parts[:-1])} import {parts[-1]}")
            else:
                __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module}")
            if install_name not in missing_deps:
                missing_deps.append(install_name)
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {' '.join(missing_deps)}")
        print(f"   Install with: pip install {' '.join(missing_deps)}")
        print("   Continuing anyway...\n")
    else:
        print("\n✓ All dependencies available\n")
    
    # Use the integrated server
    cmd = [
        sys.executable, "-m", "uvicorn",
        "mcp_metacognitive.server_integrated:app",
        "--host", "0.0.0.0",
        "--port", "8100",
        "--log-level", "info"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nStarting FULL MCP server...")
    print("This will show mcp_available: True if FastMCP loads")
    print("\nExpected endpoints:")
    print("  - http://localhost:8100/health")
    print("  - http://localhost:8100/api/system/status")
    print("  - http://localhost:8100/sse")
    print("  - http://localhost:8100/tools")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        # Run server with environment
        result = subprocess.run(cmd, env=env)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\nShutting down MCP server...")
        return 0
    except Exception as e:
        print(f"\nERROR: Failed to start MCP server: {e}")
        return 1

if __name__ == "__main__":
    # First, stop any existing MCP process
    print("Checking for existing MCP processes...")
    try:
        subprocess.run("taskkill /F /IM python.exe /FI \"WINDOWTITLE eq *8100*\"", 
                      shell=True, capture_output=True)
    except:
        pass
    
    sys.exit(main())
