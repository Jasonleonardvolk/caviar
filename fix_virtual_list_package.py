#!/usr/bin/env python3
"""
Fix the svelte-virtual import issue by finding and using the correct package
"""

import subprocess
import json
import sys
from pathlib import Path

def check_npm_package(package_name):
    """Check if a package exists on npm"""
    try:
        result = subprocess.run(
            ['npm', 'view', package_name, 'version'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except:
        return None

def main():
    print("🔍 Finding the correct Svelte virtual list package...")
    
    # Common virtual list packages for Svelte
    candidates = [
        'svelte-virtual-list',
        '@sveltejs/svelte-virtual-list', 
        'svelte-tiny-virtual-list',
        '@tanstack/svelte-virtual',
        'svelte-virtual-scroll-list'
    ]
    
    found = None
    for pkg in candidates:
        version = check_npm_package(pkg)
        if version:
            print(f"✅ Found: {pkg} (version {version})")
            found = pkg
            break
        else:
            print(f"❌ Not found: {pkg}")
    
    if not found:
        print("\n❌ No suitable virtual list package found!")
        print("\n💡 The import 'svelte-virtual' doesn't exist as a package.")
        print("   You need to either:")
        print("   1. Use a different virtual list package")
        print("   2. Remove the virtual list functionality")
        return 1
    
    print(f"\n📦 Installing {found}...")
    
    # Update package.json
    package_json_path = Path("tori_ui_svelte/package.json")
    with open(package_json_path, 'r') as f:
        package_data = json.load(f)
    
    # Remove the incorrect dependency
    if 'svelte-virtual' in package_data.get('dependencies', {}):
        del package_data['dependencies']['svelte-virtual']
    
    # Save updated package.json
    with open(package_json_path, 'w') as f:
        json.dump(package_data, f, indent=2)
    
    # Install the correct package
    subprocess.run(['npm', 'install', found], cwd='tori_ui_svelte')
    
    print(f"\n✅ Installed {found}")
    print("\n⚠️  NOTE: You'll need to update ChatFeed.svelte to import from the correct package:")
    print(f"   Change: import VirtualList from 'svelte-virtual';")
    print(f"   To:     import VirtualList from '{found}';")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
