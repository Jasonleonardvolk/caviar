#!/usr/bin/env python3
"""
Check for duplicate endpoints in the FastAPI app
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up environment
import os
os.environ["TORI_ENV"] = "test"

print("🔍 Checking for duplicate endpoints...\n")

try:
    # Import the app
    from prajna.api.prajna_api import app
    from fastapi.routing import APIRoute
    
    # Track all endpoints
    endpoints = {}
    
    for route in app.routes:
        if isinstance(route, APIRoute):
            # Create a key from path and methods
            for method in route.methods:
                key = f"{method} {route.path}"
                endpoint_name = route.endpoint.__name__ if hasattr(route.endpoint, '__name__') else str(route.endpoint)
                
                if key not in endpoints:
                    endpoints[key] = []
                endpoints[key].append({
                    'name': endpoint_name,
                    'endpoint': route.endpoint,
                    'module': route.endpoint.__module__ if hasattr(route.endpoint, '__module__') else 'unknown'
                })
    
    # Find duplicates
    duplicates = {k: v for k, v in endpoints.items() if len(v) > 1}
    
    if duplicates:
        print("⚠️ FOUND DUPLICATE ENDPOINTS:")
        for key, handlers in duplicates.items():
            print(f"\n   {key}:")
            for h in handlers:
                print(f"      - {h['name']} (from {h['module']})")
    else:
        print("✅ No duplicate endpoints found!")
    
    # Show all soliton endpoints
    print("\n📋 All Soliton endpoints:")
    soliton_endpoints = {k: v for k, v in endpoints.items() if '/soliton' in k}
    for key, handlers in soliton_endpoints.items():
        print(f"   {key} -> {handlers[0]['name']}")
    
except Exception as e:
    print(f"❌ Error checking endpoints: {e}")
    import traceback
    traceback.print_exc()
