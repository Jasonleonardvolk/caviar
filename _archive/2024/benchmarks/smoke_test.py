#!/usr/bin/env python3
"""Smoke test for soliton routes"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

print("=== Testing Soliton Route Import ===")
try:
    # Test the actual import path used by prajna_api.py
    from api.routes.soliton import router as soliton_router
    print("✅ SUCCESS: Imported from api.routes.soliton")
    print(f"   Router prefix: {soliton_router.prefix}")
    print(f"   Number of routes: {len(soliton_router.routes)}")
    for route in soliton_router.routes:
        print(f"   - {list(route.methods)[0] if route.methods else 'GET'} {soliton_router.prefix}{route.path}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Testing FastAPI App Routes ===")
try:
    # Import the actual app
    from prajna.api.prajna_api import app
    soliton_routes = [r for r in app.routes if hasattr(r, 'path') and '/soliton' in r.path]
    if soliton_routes:
        print(f"✅ Found {len(soliton_routes)} soliton routes in app:")
        for r in soliton_routes:
            print(f"   - {r.path}")
    else:
        print("⚠️  No soliton routes found in app")
except Exception as e:
    print(f"❌ FAILED to check app routes: {e}")

print("\n=== Testing MCP Signature ===")
try:
    import inspect
    # Try to import the MCP module
    sys.path.insert(0, str(Path(__file__).parent / "mcp_metacognitive"))
    try:
        from fastmcp import FastMCP
        mcp = FastMCP("test")
        sig = inspect.signature(mcp.run)
        print(f"✅ FastMCP.run signature: {sig}")
        params = list(sig.parameters.keys())
        if 'address' in params:
            print("   ⚠️  WARNING: This version expects 'address' parameter!")
        elif 'host' in params and 'port' in params:
            print("   ✅ This version expects 'host' and 'port' parameters (old signature)")
        else:
            print(f"   ❓ Unexpected parameters: {params}")
    except ImportError:
        print("   ℹ️  FastMCP not available in this environment")
except Exception as e:
    print(f"❌ FAILED to check MCP: {e}")

print("\n=== Summary ===")
print("Run this after starting the API to verify routes are mounted.")
print("If soliton routes are missing, check the startup logs for import errors.")
