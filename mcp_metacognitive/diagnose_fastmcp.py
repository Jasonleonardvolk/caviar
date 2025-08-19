#!/usr/bin/env python3
"""
Diagnose FastMCP app exposure
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from mcp.server.fastmcp import FastMCP
    
    # Create instance
    mcp = FastMCP(name="test", version="1.0")
    
    print("FastMCP instance created")
    print(f"Type: {type(mcp)}")
    print(f"Dir: {[attr for attr in dir(mcp) if not attr.startswith('_')]}")
    
    # Check for app attributes
    for attr in ['app', 'application', 'asgi', 'server', 'fastapi_app']:
        if hasattr(mcp, attr):
            val = getattr(mcp, attr)
            print(f"\nFound {attr}: {type(val)}")
            
    # Check if it's an ASGI app itself
    if hasattr(mcp, '__call__'):
        print("\nFastMCP is callable")
        
    # Try to find the FastAPI app
    import inspect
    for name, obj in inspect.getmembers(mcp):
        if 'fastapi' in str(type(obj)).lower() or 'app' in str(type(obj)).lower():
            print(f"\nPotential app found: {name} = {type(obj)}")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
