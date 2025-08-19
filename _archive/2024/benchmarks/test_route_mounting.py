#!/usr/bin/env python3
"""Quick test to see if routes are actually mounted"""

import os
import sys
from pathlib import Path

# Setup environment
os.environ["TORI_ENV"] = "test"
sys.path.insert(0, str(Path(__file__).parent))

print("Testing route mounting...\n")

# Test 1: Direct router import
try:
    from api.routes.soliton import router as soliton_router
    print(f"✅ Soliton router imported: {soliton_router.prefix}")
    print(f"   Routes: {[r.path for r in soliton_router.routes]}")
except Exception as e:
    print(f"❌ Router import failed: {e}")

# Test 2: Check if it mounts to a test app
print("\nTesting FastAPI mounting...")
try:
    from fastapi import FastAPI
    test_app = FastAPI()
    test_app.include_router(soliton_router)
    
    routes = [r.path for r in test_app.routes if hasattr(r, 'path') and '/soliton' in r.path]
    print(f"✅ Mounted to test app: {len(routes)} soliton routes")
    for r in routes:
        print(f"   - {r}")
except Exception as e:
    print(f"❌ Mount test failed: {e}")

# Test 3: Check prajna_api startup
print("\nChecking prajna_api startup event...")
try:
    # We need to trigger the startup event
    from prajna.api.prajna_api import app, load_prajna_model, SOLITON_ROUTES_AVAILABLE
    
    print(f"SOLITON_ROUTES_AVAILABLE = {SOLITON_ROUTES_AVAILABLE}")
    
    # Check current routes before startup
    before_routes = [r.path for r in app.routes if hasattr(r, 'path') and '/soliton' in r.path]
    print(f"Routes before startup: {len(before_routes)}")
    
    # Run the startup event
    import asyncio
    asyncio.run(load_prajna_model())
    
    # Check routes after startup
    after_routes = [r.path for r in app.routes if hasattr(r, 'path') and '/soliton' in r.path]
    print(f"Routes after startup: {len(after_routes)}")
    
    if after_routes:
        print("✅ Soliton routes mounted!")
        for r in after_routes:
            print(f"   - {r}")
    else:
        print("❌ No soliton routes found after startup!")
        
except Exception as e:
    print(f"❌ Startup test failed: {e}")
    import traceback
    traceback.print_exc()
