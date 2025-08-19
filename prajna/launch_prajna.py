#!/usr/bin/env python3
"""
Launch Prajna with Correct Imports
===================================

Ensures Python path is set up correctly before launching Prajna.
"""

import sys
import os
from pathlib import Path

# Get the repo root (parent of kha directory)
script_dir = Path(__file__).resolve().parent
kha_dir = script_dir.parent  # prajna -> kha
repo_root = kha_dir.parent   # kha -> tori (repo root)

# Add repo root to Python path so "kha.api" imports work
sys.path.insert(0, str(repo_root))

print(f"Repo root: {repo_root}")
print(f"Python path includes: {sys.path[0]}")

# Now import and run Prajna
try:
    print("\nImporting Prajna API...")
    from prajna_api import app
    
    print("Testing kha.api import...")
    from kha.api import app as main_api
    print("✅ kha.api import successful!")
    
    # Run Prajna
    import uvicorn
    print("\nStarting Prajna API server on http://localhost:8002")
    print("Documentation at: http://localhost:8002/docs")
    uvicorn.run("prajna_api:app", host="0.0.0.0", port=8002, reload=True)
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("\nMake sure you have:")
    print("  1. kha/api/__init__.py with: from .main import app")
    print("  2. kha/api/main.py with FastAPI app")
    print("  3. Run this script from kha/prajna/ directory")
