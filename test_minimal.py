#!/usr/bin/env python3
"""
Test minimal soliton route mounting
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Minimal test
from fastapi import FastAPI
from api.routes.soliton import router

app = FastAPI()
app.include_router(router)

# Print all routes
print("Routes in app:")
for route in app.routes:
    if hasattr(route, 'path'):
        print(f"  {route.path}")

# Filter soliton routes
soliton_routes = [r for r in app.routes if hasattr(r, 'path') and '/soliton' in r.path]
print(f"\nSoliton routes: {len(soliton_routes)}")
for r in soliton_routes:
    print(f"  {r.path}")
