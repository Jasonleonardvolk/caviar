#!/usr/bin/env python3
"""
MCP Service Setup Verifier
Checks that MCP is properly configured and can be imported
"""

import os
import sys
from pathlib import Path

def check_mcp_setup():
    """Check MCP setup and configuration"""
    print("=== MCP Service Setup Verification ===\n")
    
    # Check current directory
    cwd = Path.cwd()
    print(f"1. Current directory: {cwd}")
    
    # Check if we're in the right place
    expected_dir = Path("C:/Users/jason/Desktop/tori/kha")
    if cwd != expected_dir:
        print(f"   WARNING: Not in expected directory {expected_dir}")
        print(f"   Run this from: cd {expected_dir}")
    else:
        print("   ✓ Correct directory")
    
    # Check Python path
    print(f"\n2. Python path:")
    for i, path in enumerate(sys.path):
        print(f"   [{i}] {path}")
    
    if str(cwd) in sys.path[:3]:
        print("   ✓ Current directory is in Python path")
    else:
        print("   WARNING: Current directory not in Python path")
    
    # Check mcp_metacognitive directory
    print("\n3. Checking mcp_metacognitive directory:")
    mcp_dir = cwd / "mcp_metacognitive"
    if mcp_dir.exists():
        print(f"   ✓ Directory exists: {mcp_dir}")
        
        # Check for __init__.py
        init_file = mcp_dir / "__init__.py"
        if init_file.exists():
            print("   ✓ __init__.py exists")
        else:
            print("   ✗ __init__.py missing - creating it...")
            init_file.touch()
            print("   ✓ Created __init__.py")
        
        # Check for server files
        server_files = [
            "server_simple_fixed.py",
            "server_simple.py", 
            "server_fixed.py",
            "server.py"
        ]
        
        print("\n4. Checking server files:")
        found_servers = []
        for server_file in server_files:
            server_path = mcp_dir / server_file
            if server_path.exists():
                print(f"   ✓ {server_file} exists")
                found_servers.append(server_file)
            else:
                print(f"   ✗ {server_file} not found")
        
        if not found_servers:
            print("   ERROR: No server files found!")
    else:
        print(f"   ✗ Directory not found: {mcp_dir}")
    
    # Try imports
    print("\n5. Testing imports:")
    
    # Add current directory to path if needed
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))
        print(f"   Added {cwd} to sys.path")
    
    try:
        import mcp_metacognitive
        print("   ✓ Can import mcp_metacognitive package")
    except ImportError as e:
        print(f"   ✗ Cannot import mcp_metacognitive: {e}")
    
    try:
        from mcp_metacognitive import server_simple_fixed
        print("   ✓ Can import server_simple_fixed")
    except ImportError as e:
        print(f"   ✗ Cannot import server_simple_fixed: {e}")
    
    # Check for dependencies
    print("\n6. Checking dependencies:")
    deps = ["fastapi", "uvicorn", "sse-starlette"]
    missing_deps = []
    
    for dep in deps:
        try:
            __import__(dep.replace("-", "_"))
            print(f"   ✓ {dep} installed")
        except ImportError:
            print(f"   ✗ {dep} not installed")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n   To install missing dependencies:")
        print(f"   pip install {' '.join(missing_deps)}")
    
    # Port assignment reminder
    print("\n7. Port assignments:")
    print("   - MCP Server: Port 8100")
    print("   - Main API (with /api/health): Port 8002")
    print("   - Frontend: Port 5173")
    print("   - Audio Bridge: Port 8765")
    print("   - Concept Mesh Bridge: Port 8766")
    
    print("\n=== Summary ===")
    if found_servers and not missing_deps:
        print("✓ MCP setup looks good!")
        print("\nTo start MCP:")
        print("  python start_mcp_manual_fixed.py")
    else:
        print("✗ Issues found - fix them before starting MCP")

if __name__ == "__main__":
    check_mcp_setup()
