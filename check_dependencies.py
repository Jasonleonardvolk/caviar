#!/usr/bin/env python3
"""Check all TORI dependencies and report status"""

import subprocess
import sys
import json
from pathlib import Path

def check_npm_package(package_name, package_dir):
    """Check if an npm package is installed"""
    try:
        package_json = package_dir / "node_modules" / package_name / "package.json"
        if package_json.exists():
            with open(package_json) as f:
                data = json.load(f)
                return True, data.get('version', 'unknown')
        return False, None
    except:
        return False, None

def main():
    print("🔍 TORI DEPENDENCY CHECK")
    print("=" * 60)
    
    frontend_dir = Path(__file__).parent / "tori_ui_svelte"
    
    # Critical frontend dependencies
    critical_deps = [
        'svelte',
        '@sveltejs/kit',
        'vite',
        'svelte-virtual',
        '@tailwindcss/postcss',
        'tailwindcss',
        'mathjs'
    ]
    
    print("\n📦 Frontend Dependencies:")
    all_good = True
    for dep in critical_deps:
        installed, version = check_npm_package(dep, frontend_dir)
        if installed:
            print(f"  ✅ {dep} v{version}")
        else:
            print(f"  ❌ {dep} - NOT INSTALLED")
            all_good = False
    
    # Python dependencies
    print("\n🐍 Python Dependencies:")
    python_deps = ['psutil', 'requests', 'uvicorn', 'websockets', 'asyncio']
    
    for dep in python_deps:
        try:
            __import__(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep} - NOT INSTALLED")
            all_good = False
    
    if all_good:
        print("\n✅ All dependencies are installed!")
    else:
        print("\n❌ Some dependencies are missing!")
        print("\n💡 To fix:")
        print("   1. cd tori_ui_svelte && npm install")
        print("   2. pip install -r requirements.txt")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
