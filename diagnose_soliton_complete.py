#!/usr/bin/env python3
"""
COMPREHENSIVE SOLITON ROUTE DIAGNOSTIC
=====================================
This script tests every aspect of the soliton route setup
"""

import os
import sys
import json
import subprocess
import time
import requests
from pathlib import Path

# Setup
sys.path.insert(0, str(Path(__file__).parent))
os.environ["TORI_ENV"] = "test"

print("🔍 SOLITON ROUTE DIAGNOSTIC TOOL")
print("=" * 50)

# Test 1: File Structure
print("\n1. FILE STRUCTURE CHECK:")
files_to_check = [
    "api/routes/soliton.py",
    "api/main.py",
    "api/soliton.py.deprecated",
    "api/soliton_route.py",
    "prajna/api/prajna_api.py",
    "enhanced_launcher.py"
]

for file in files_to_check:
    path = Path(file)
    if path.exists():
        status = "✅ EXISTS"
        if "deprecated" in str(path) or file == "api/main.py":
            status += " (DEPRECATED)"
    else:
        status = "❌ MISSING"
    print(f"   {file}: {status}")

# Test 2: Import Chain
print("\n2. IMPORT CHAIN TEST:")
try:
    # Test the actual import used by prajna_api
    from api.routes.soliton import router as soliton_router
    print("   ✅ api.routes.soliton imports successfully")
    print(f"      Prefix: {soliton_router.prefix}")
    print(f"      Routes: {len(soliton_router.routes)}")
except Exception as e:
    print(f"   ❌ Import failed: {e}")

# Test 3: Field Name Check
print("\n3. FIELD NAME CHECK:")
try:
    from api.routes.soliton import SolitonInitRequest
    import inspect
    
    # Get field names
    fields = SolitonInitRequest.__fields__
    print("   SolitonInitRequest fields:")
    for field_name, field_info in fields.items():
        print(f"      - {field_name}: {field_info.type_}")
    
    if "user_id" in fields:
        print("   ✅ Has 'user_id' field (matches frontend)")
    elif "userId" in fields:
        print("   ❌ Has 'userId' field (mismatch with frontend)")
    else:
        print("   ❌ Missing expected field")
        
except Exception as e:
    print(f"   ❌ Field check failed: {e}")

# Test 4: Prajna API Configuration
print("\n4. PRAJNA API CONFIGURATION:")
try:
    from prajna.api.prajna_api import (
        SOLITON_ROUTES_AVAILABLE,
        soliton_router as imported_router
    )
    
    print(f"   SOLITON_ROUTES_AVAILABLE = {SOLITON_ROUTES_AVAILABLE}")
    
    if imported_router:
        print("   ✅ soliton_router imported in prajna_api")
    else:
        print("   ❌ soliton_router is None in prajna_api")
        
except Exception as e:
    print(f"   ❌ Prajna API check failed: {e}")

# Test 5: Live Server Test
print("\n5. LIVE SERVER TEST:")
print("   Checking if API server is running...")

try:
    response = requests.get("http://localhost:8002/api/health", timeout=2)
    if response.status_code == 200:
        print("   ✅ API server is running")
        
        # Test soliton endpoints
        print("\n   Testing Soliton endpoints:")
        
        # Test init endpoint
        init_data = {"user_id": "test_user"}
        print(f"   POST /api/soliton/init with {json.dumps(init_data)}")
        
        init_response = requests.post(
            "http://localhost:8002/api/soliton/init",
            json=init_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"   Status: {init_response.status_code}")
        if init_response.status_code == 200:
            print(f"   ✅ Init endpoint working: {init_response.json()}")
        elif init_response.status_code == 422:
            print(f"   ❌ Validation error: {init_response.text}")
        elif init_response.status_code == 404:
            print(f"   ❌ Route not found! Routes not mounted properly")
        else:
            print(f"   ❌ Error: {init_response.text}")
            
        # Test stats endpoint
        print("\n   GET /api/soliton/stats/testuser")
        stats_response = requests.get("http://localhost:8002/api/soliton/stats/testuser")
        
        print(f"   Status: {stats_response.status_code}")
        if stats_response.status_code == 200:
            print(f"   ✅ Stats endpoint working: {stats_response.json()}")
        else:
            print(f"   ❌ Error: {stats_response.text}")
            
        # Check OpenAPI
        print("\n   Checking OpenAPI documentation:")
        openapi_response = requests.get("http://localhost:8002/openapi.json")
        if openapi_response.status_code == 200:
            openapi = openapi_response.json()
            paths = openapi.get("paths", {})
            soliton_paths = [p for p in paths if "/soliton" in p]
            
            if soliton_paths:
                print(f"   ✅ Found {len(soliton_paths)} soliton routes in OpenAPI")
                for path in soliton_paths:
                    print(f"      - {path}")
            else:
                print("   ❌ No soliton routes in OpenAPI!")
                
    else:
        print(f"   ❌ API server not running (status {response.status_code})")
        print("   Run: python enhanced_launcher.py")
        
except requests.exceptions.ConnectionError:
    print("   ❌ API server not running")
    print("   Run: python enhanced_launcher.py")
except Exception as e:
    print(f"   ❌ Server test error: {e}")

# Summary
print("\n" + "=" * 50)
print("DIAGNOSTIC SUMMARY:")
print("=" * 50)

print("\nIf soliton routes are 404ing, check:")
print("1. Is the API server running? (python enhanced_launcher.py)")
print("2. Check the startup logs for import errors")
print("3. Make sure api/routes/soliton.py exists and is valid")
print("4. Check that prajna_api.py includes the router in the startup event")
print("5. Verify the field names match between frontend and backend")

print("\nTo fix common issues:")
print("- 404 errors: Routes not mounted, check startup logs")
print("- 422 errors: Field name mismatch (user_id vs userId)")
print("- 500 errors: Backend errors, check server logs")
