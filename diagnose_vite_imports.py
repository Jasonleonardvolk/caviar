#!/usr/bin/env python3
"""Diagnose and fix Vite import issues in the holographic system"""

import re
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def diagnose_imports():
    print("🔍 DIAGNOSING VITE IMPORT ISSUES")
    print("=" * 70)
    
    base_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib")
    
    # Files that might have issues
    suspect_files = [
        "realGhostEngine.js",
        "realGhostEngine_v2.js",
        "conceptHologramRenderer.js",
        "components/HolographicDisplay.svelte"
    ]
    
    issues_found = []
    
    for file_rel in suspect_files:
        file_path = base_path / file_rel
        if not file_path.exists():
            continue
            
        print(f"\n📄 Checking {file_rel}...")
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Check for common issues
            
            # 1. TypeScript imports with .js extension
            ts_imports = re.findall(r"import\s+.*?\s+from\s+['\"]([^'\"]+\.js)['\"]", content)
            for imp in ts_imports:
                if '../../../frontend/lib' in imp:
                    print(f"  ⚠️ TypeScript import with .js: {imp}")
                    issues_found.append(("ts_import_js", file_path, imp))
            
            # 2. JSX syntax in .js file
            if file_path.suffix == '.js':
                jsx_patterns = [
                    r'<[A-Z][a-zA-Z0-9]*[\s/>]',  # <Component
                    r'</[A-Z][a-zA-Z0-9]*>',       # </Component>
                    r'return\s*\(',                 # return (
                ]
                for pattern in jsx_patterns:
                    if re.search(pattern, content):
                        print(f"  ⚠️ Possible JSX syntax in .js file")
                        issues_found.append(("jsx_in_js", file_path, pattern))
            
            # 3. TypeScript syntax in .js file
            if file_path.suffix == '.js':
                ts_patterns = [
                    r':\s*(string|number|boolean|any|void)',  # Type annotations
                    r'interface\s+\w+',                        # Interface
                    r'type\s+\w+\s*=',                        # Type alias
                    r'<[A-Za-z]+>',                           # Generic syntax
                ]
                for pattern in ts_patterns:
                    if re.search(pattern, content):
                        print(f"  ⚠️ TypeScript syntax in .js file: {pattern}")
                        issues_found.append(("ts_in_js", file_path, pattern))
            
            # 4. Missing .svelte extension
            svelte_imports = re.findall(r"import\s+\w+\s+from\s+['\"]([^'\"]+)['\"]", content)
            for imp in svelte_imports:
                if '/components/' in imp and not imp.endswith('.svelte') and not imp.endswith('.js') and not imp.endswith('.ts'):
                    print(f"  ⚠️ Possible missing .svelte extension: {imp}")
                    issues_found.append(("missing_svelte_ext", file_path, imp))
            
        except Exception as e:
            print(f"  ❌ Error reading file: {e}")
    
    # Check HolographicDisplay.svelte specifically
    holo_display = base_path / "components" / "HolographicDisplay.svelte"
    if holo_display.exists():
        print(f"\n📄 Checking HolographicDisplay.svelte...")
        content = holo_display.read_text(encoding='utf-8')
        
        # Check if it's importing RealGhostEngine correctly
        if "from '$lib/realGhostEngine.js'" in content:
            print("  ✅ Importing realGhostEngine.js correctly")
        elif "from '../realGhostEngine'" in content:
            print("  ⚠️ Using relative import instead of $lib alias")
            issues_found.append(("relative_import", holo_display, "realGhostEngine"))
    
    print("\n\n📊 SUMMARY:")
    print("-" * 70)
    print(f"Found {len(issues_found)} potential issues")
    
    if issues_found:
        print("\n🔧 RECOMMENDED FIXES:")
        
        # Group by issue type
        ts_import_issues = [i for i in issues_found if i[0] == "ts_import_js"]
        if ts_import_issues:
            print("\n1. TypeScript imports with .js extension:")
            print("   Option A: Remove .js extension from TypeScript imports")
            print("   Option B: Configure Vite to handle .js → .ts resolution")
            print("   Option C: Use the actual .ts extension")
        
        jsx_issues = [i for i in issues_found if i[0] == "jsx_in_js"]
        if jsx_issues:
            print("\n2. JSX syntax in .js files:")
            print("   - Rename file to .jsx")
            print("   - Or remove JSX syntax")
        
        ts_syntax_issues = [i for i in issues_found if i[0] == "ts_in_js"]
        if ts_syntax_issues:
            print("\n3. TypeScript syntax in .js files:")
            print("   - Remove type annotations")
            print("   - Or rename file to .ts")
    
    return issues_found

def create_import_fix():
    """Create a fix that removes .js extensions from TypeScript imports"""
    
    print("\n\n🔧 CREATING IMPORT FIX")
    print("=" * 70)
    
    fix_content = '''// RealGhostEngine.js - Fixed imports for Vite
// NOW WITH CONCEPT MESH INTEGRATION!

// Import from TypeScript files WITHOUT .js extension
// Vite will handle the resolution correctly
import HolographicEngine from '../../../frontend/lib/holographicEngine';
import { ToriHolographicRenderer } from '../../../frontend/lib/holographicRenderer';
import { FFTCompute } from '../../../frontend/lib/webgpu/fftCompute';
import { HologramPropagation } from '../../../frontend/lib/webgpu/hologramPropagation';
import { QuiltGenerator } from '../../../frontend/lib/webgpu/quiltGenerator';
import { ConceptHologramRenderer } from './conceptHologramRenderer';

// Rest of the file continues as before...
'''
    
    print("Fix created - remove .js extensions from TypeScript imports")
    print("\nTo apply, update the import section of realGhostEngine.js")
    
    # Also check vite.config.js
    vite_config = Path(r"{PROJECT_ROOT}\tori_ui_svelte\vite.config.js")
    if vite_config.exists():
        print("\n📄 Checking vite.config.js...")
        content = vite_config.read_text(encoding='utf-8')
        if "resolve:" in content and "extensions:" in content:
            print("  ✅ Vite config has resolve.extensions")
        else:
            print("  ⚠️ May need to add resolve.extensions to vite.config.js")

if __name__ == "__main__":
    issues = diagnose_imports()
    create_import_fix()
    
    print("\n\n🎯 NEXT STEPS:")
    print("1. Run: npx vite --debug")
    print("2. Look for the exact file causing the error")
    print("3. Apply the appropriate fix from above")
    print("4. Restart Vite")
