#!/usr/bin/env python3
"""
Test API Import Chain
=====================

Verify that the api.main module can be imported correctly.
"""

import sys
from pathlib import Path

# Add kha directory to Python path
kha_path = Path(__file__).parent
sys.path.insert(0, str(kha_path))

print("Python path:")
for p in sys.path[:3]:
    print(f"  {p}")

print("\nTesting imports...")

try:
    print("1. Importing api package...")
    import api
    print("   ✓ api package imported")
    
    print("2. Importing api.main...")
    import api.main
    print("   ✓ api.main imported")
    
    print("3. Importing app from api...")
    from api import app
    print(f"   ✓ app imported: {app}")
    
    print("4. Importing routers...")
    from api import soliton_router, concept_mesh_router
    print("   ✓ routers imported")
    
    print("\n✅ All imports successful!")
    
    # Show available endpoints
    print("\nAvailable routes in app:")
    for route in app.routes:
        if hasattr(route, 'path'):
            print(f"  {route.methods} {route.path}")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("\nDebug info:")
    print(f"  Current directory: {Path.cwd()}")
    print(f"  Script location: {Path(__file__).parent}")
    
    # Check if files exist
    api_path = kha_path / "api"
    print(f"\n  api directory exists: {api_path.exists()}")
    if api_path.exists():
        print(f"  __init__.py exists: {(api_path / '__init__.py').exists()}")
        print(f"  main.py exists: {(api_path / 'main.py').exists()}")
        print(f"  soliton.py exists: {(api_path / 'soliton.py').exists()}")
