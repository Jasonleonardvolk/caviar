#!/usr/bin/env python3
"""
Diagnose Soliton Route Issues
=============================
This script checks if soliton routes are properly imported and mounted
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=== SOLITON ROUTE DIAGNOSTICS ===\n")

# Step 1: Check if the route file exists
print("1. Checking route file existence:")
route_file = Path("api/routes/soliton.py")
if route_file.exists():
    print(f"   ✅ {route_file} exists")
else:
    print(f"   ❌ {route_file} NOT FOUND!")

# Step 2: Try to import the router
print("\n2. Testing router import:")
try:
    from api.routes.soliton import router as soliton_router
    print("   ✅ Successfully imported soliton router")
    print(f"   Router type: {type(soliton_router)}")
    print(f"   Router prefix: {soliton_router.prefix}")
    print(f"   Number of routes: {len(soliton_router.routes)}")
    
    print("\n   Routes defined:")
    for route in soliton_router.routes:
        methods = list(route.methods) if hasattr(route, 'methods') and route.methods else ['GET']
        print(f"   - {methods[0]} {route.path}")
        
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    import traceback
    traceback.print_exc()

# Step 3: Check if prajna_api imports it
print("\n3. Checking prajna_api.py imports:")
try:
    # Read the prajna_api file to check imports
    prajna_api_path = Path("prajna/api/prajna_api.py")
    if prajna_api_path.exists():
        content = prajna_api_path.read_text()
        if "from api.routes.soliton import router as soliton_router" in content:
            print("   ✅ prajna_api.py contains soliton import")
        else:
            print("   ❌ prajna_api.py does NOT import soliton router")
            
        if "app.include_router(soliton_router)" in content:
            print("   ✅ prajna_api.py includes soliton router")
        else:
            print("   ❌ prajna_api.py does NOT include soliton router")
    else:
        print(f"   ❌ {prajna_api_path} not found")
        
except Exception as e:
    print(f"   ❌ Error checking prajna_api: {e}")

# Step 4: Try to import the full app and check routes
print("\n4. Checking final app routes:")
try:
    # Set environment variables to prevent startup issues
    os.environ["TORI_ENV"] = "test"
    os.environ["PRAJNA_MODEL_TYPE"] = "saigon"
    
    from prajna.api.prajna_api import app
    
    # Count routes
    all_routes = [r for r in app.routes if hasattr(r, 'path')]
    soliton_routes = [r for r in all_routes if '/soliton' in r.path]
    
    print(f"   Total routes in app: {len(all_routes)}")
    print(f"   Soliton routes found: {len(soliton_routes)}")
    
    if soliton_routes:
        print("\n   Soliton endpoints:")
        for route in soliton_routes:
            print(f"   - {route.path}")
    else:
        print("\n   ❌ NO SOLITON ROUTES FOUND IN APP!")
        print("\n   Available routes (first 20):")
        for route in all_routes[:20]:
            print(f"   - {route.path}")
            
except Exception as e:
    print(f"   ❌ Error importing app: {e}")
    import traceback
    traceback.print_exc()

# Step 5: Check for common issues
print("\n5. Common issues check:")

# Check for SOLITON_ROUTES_AVAILABLE flag
try:
    from prajna.api.prajna_api import SOLITON_ROUTES_AVAILABLE
    print(f"   SOLITON_ROUTES_AVAILABLE = {SOLITON_ROUTES_AVAILABLE}")
except:
    print("   ❌ Could not check SOLITON_ROUTES_AVAILABLE flag")

# Check sys.path
print("\n6. Python path:")
for i, p in enumerate(sys.path[:5]):
    print(f"   [{i}] {p}")

print("\n=== DIAGNOSIS COMPLETE ===")
print("\nIf soliton routes are missing, check:")
print("1. Import errors in the console when starting the API")
print("2. The SOLITON_ROUTES_AVAILABLE flag in prajna_api.py")
print("3. Whether app.include_router() is actually called in the startup event")
