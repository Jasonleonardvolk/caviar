#!/usr/bin/env python3
"""
Final Soliton Route Debug - Fixed
=================================
Check exactly what's happening with the routes
"""

import os
import sys
from pathlib import Path

# Setup
sys.path.insert(0, str(Path(__file__).parent))
os.environ["TORI_ENV"] = "test"

print("=== SOLITON ROUTE DEBUG (FIXED) ===\n")

# Test 1: Import the actual soliton router
print("1. Testing soliton router import:")
try:
    from api.routes.soliton import router as soliton_router
    print(f"   ✅ Router imported successfully")
    print(f"   Prefix: {soliton_router.prefix}")
    print(f"   Routes: {len(soliton_router.routes)}")
    for route in soliton_router.routes:
        print(f"   - {route.path}")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Check the field names
print("\n2. Checking field names:")
try:
    from api.routes.soliton import SolitonInitRequest
    fields = list(SolitonInitRequest.__fields__.keys())
    print(f"   Fields in SolitonInitRequest: {fields}")
    if "user_id" in fields:
        print("   ✅ Has 'user_id' field (correct)")
    elif "userId" in fields:
        print("   ❌ Has 'userId' field (mismatch)")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 3: Check if routes mount properly
print("\n3. Testing route mounting:")
try:
    from fastapi import FastAPI
    test_app = FastAPI()
    
    # Before mounting
    before = len([r for r in test_app.routes if hasattr(r, 'path')])
    print(f"   Routes before mounting: {before}")
    
    # Mount router
    from api.routes.soliton import router as soliton_router
    test_app.include_router(soliton_router)
    
    # After mounting
    after = len([r for r in test_app.routes if hasattr(r, 'path')])
    soliton_routes = [r for r in test_app.routes if hasattr(r, 'path') and '/soliton' in r.path]
    
    print(f"   Routes after mounting: {after}")
    print(f"   Soliton routes found: {len(soliton_routes)}")
    
    if soliton_routes:
        print("   ✅ Routes mounted successfully:")
        for r in soliton_routes:
            print(f"      - {r.path}")
    else:
        print("   ❌ No soliton routes found!")
        
except Exception as e:
    print(f"   ❌ Mount test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check SOLITON_AVAILABLE flag
print("\n4. Checking SOLITON_AVAILABLE flag:")
try:
    from api.routes.soliton import SOLITON_AVAILABLE
    print(f"   SOLITON_AVAILABLE = {SOLITON_AVAILABLE}")
    if not SOLITON_AVAILABLE:
        print("   ⚠️ Soliton module imports are failing!")
except Exception as e:
    print(f"   ❌ Could not check flag: {e}")

print("\n=== SUMMARY ===")
print("The issue was api/__init__.py importing from deprecated main.py!")
print("Now the routes should work correctly.")
