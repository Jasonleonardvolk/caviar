#!/usr/bin/env python3
"""
Script to ensure all required __init__.py files exist for the full API
"""

import os
from pathlib import Path

def create_init_files():
    """Create missing __init__.py files for prajna package structure"""
    
    script_dir = Path(__file__).parent
    
    # Define required __init__.py files
    init_files = [
        script_dir / "prajna" / "__init__.py",
        script_dir / "prajna" / "api" / "__init__.py",
    ]
    
    created = []
    already_exists = []
    
    for init_file in init_files:
        # Create parent directory if it doesn't exist
        init_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not init_file.exists():
            # Create empty __init__.py file
            init_file.write_text("# Auto-generated __init__.py\n")
            created.append(init_file)
            print(f"✅ Created: {init_file.relative_to(script_dir)}")
        else:
            already_exists.append(init_file)
            print(f"📁 Already exists: {init_file.relative_to(script_dir)}")
    
    print("\n📋 Summary:")
    print(f"   Created: {len(created)} files")
    print(f"   Already existed: {len(already_exists)} files")
    
    # Check if prajna_api.py exists
    api_file = script_dir / "prajna" / "api" / "prajna_api.py"
    if api_file.exists():
        print(f"\n✅ Full API file found: {api_file.relative_to(script_dir)}")
    else:
        print(f"\n❌ Full API file NOT FOUND: {api_file.relative_to(script_dir)}")
        print("   You need to create this file with the full FastAPI application")
        print("\n📝 The prajna_api.py file should contain:")
        print("   from fastapi import FastAPI")
        print("   app = FastAPI()")
        print("   # ...routes for concept mesh, soliton memory, etc...")

if __name__ == "__main__":
    print("🔧 Checking prajna package structure...")
    create_init_files()
