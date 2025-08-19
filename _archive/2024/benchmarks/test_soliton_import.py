#!/usr/bin/env python3
"""Test script to verify soliton route imports"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing soliton route imports...")

try:
    from api.routes.soliton import router as soliton_router
    print("✅ SUCCESS: Imported soliton router from api.routes.soliton")
    print(f"   Router type: {type(soliton_router)}")
    print(f"   Router prefix: {soliton_router.prefix}")
    print(f"   Routes: {len(soliton_router.routes)}")
    for route in soliton_router.routes:
        print(f"   - {route.methods} {route.path}")
except ImportError as e:
    print(f"❌ FAILED: Could not import soliton router")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting if router mounts correctly...")
try:
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(soliton_router)
    print("✅ SUCCESS: Router mounted to FastAPI app")
    
    # List all routes
    print("\nAll routes in app:")
    for route in app.routes:
        if hasattr(route, 'path'):
            print(f"   {route.path}")
            
except Exception as e:
    print(f"❌ FAILED: Could not mount router")
    print(f"   Error: {e}")
