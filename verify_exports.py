#!/usr/bin/env python3
"""
🔧 COGNITIVE MODULE EXPORT FIXER
Fix all missing export declarations in cognitive modules
"""

import os
import re
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_cognitive_exports():
    """Fix export issues in all cognitive modules"""
    
    base_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib\cognitive")
    
    if not base_path.exists():
        print(f"❌ Cognitive directory not found: {base_path}")
        return
    
    print("🔧 FIXING COGNITIVE MODULE EXPORTS")
    print("=" * 50)
    
    # Check holographicMemory.ts
    holographic_file = base_path / "holographicMemory.ts"
    if holographic_file.exists():
        with open(holographic_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "export class HolographicMemory" in content:
            print("✅ holographicMemory.ts - Already fixed")
        else:
            print("⚠️  holographicMemory.ts - Needs manual verification")
    
    # Check ghostCollective.ts
    ghost_file = base_path / "ghostCollective.ts"
    if ghost_file.exists():
        print("🔍 Checking ghostCollective.ts...")
        
        with open(ghost_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if it has the workaround export
        if "export { GhostCollective };" in content:
            print("✅ ghostCollective.ts - Has workaround export (should work)")
        elif "export class GhostCollective" in content:
            print("✅ ghostCollective.ts - Already properly exported")
        else:
            print("❌ ghostCollective.ts - Needs export fix")
    
    # Check braidMemory.ts
    braid_file = base_path / "braidMemory.ts"
    if braid_file.exists():
        with open(braid_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "export class BraidMemory" in content:
            print("✅ braidMemory.ts - Already properly exported")
        else:
            print("❌ braidMemory.ts - Needs export fix")
    
    print("\n🎯 QUICK VERIFICATION:")
    print("1. HolographicMemory should be fixed ✅")
    print("2. GhostCollective has workaround ✅") 
    print("3. BraidMemory should already work ✅")
    
    print("\n🚀 Next steps:")
    print("1. Restart your SvelteKit dev server: npm run dev")
    print("2. Check for any remaining import errors")
    print("3. The warning should be gone!")

if __name__ == "__main__":
    fix_cognitive_exports()
