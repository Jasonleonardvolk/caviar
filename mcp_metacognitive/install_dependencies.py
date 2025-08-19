#!/usr/bin/env python3
"""
Requirements Installation Script for TORI MCP Metacognitive Server
================================================================

This script installs the required dependencies with fallback options.
"""

import subprocess
import sys
import os

def run_command(cmd, check=True):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    print("🚀 TORI MCP Metacognitive Server - Dependency Installation")
    print("=" * 60)
    
    # Core dependencies that should always be installed
    core_deps = [
        "numpy>=1.24.0",
        "scipy>=1.10.0", 
        "scikit-learn>=1.3.0",
        "aiofiles>=23.0.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "httpx>=0.25.0",
        "uvicorn>=0.24.0",
        "fastapi>=0.104.0",
    ]
    
    # Optional MCP dependencies
    mcp_deps = [
        "mcp-server-sdk",
        "fastmcp",
    ]
    
    print("📦 Installing core dependencies...")
    for dep in core_deps:
        print(f"  Installing {dep}...")
        if not run_command(f"{sys.executable} -m pip install '{dep}'"):
            print(f"  ⚠️  Failed to install {dep}, continuing...")
    
    print("\n📦 Attempting to install MCP packages...")
    mcp_installed = False
    
    # Try to install MCP packages
    for dep in mcp_deps:
        print(f"  Installing {dep}...")
        if run_command(f"{sys.executable} -m pip install '{dep}'", check=False):
            mcp_installed = True
        else:
            print(f"  ℹ️  {dep} not available - will use fallback implementation")
    
    # If MCP packages failed, ensure FastAPI is available as fallback
    if not mcp_installed:
        print("\n📦 Ensuring FastAPI fallback is available...")
        run_command(f"{sys.executable} -m pip install fastapi uvicorn")
    
    print("\n✅ Installation complete!")
    print("\n📋 Summary:")
    print("  - Core dependencies: Installed")
    if mcp_installed:
        print("  - MCP packages: Installed")
        print("  - Server mode: FastMCP (native)")
    else:
        print("  - MCP packages: Not available")
        print("  - Server mode: FastAPI (fallback)")
    
    print("\n🚀 You can now run the server with:")
    print("  python server.py")
    print("  or")
    print("  python run_production_server.py")

if __name__ == "__main__":
    main()
