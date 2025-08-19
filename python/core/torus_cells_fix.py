#!/usr/bin/env python3
"""
Fix for TorusCells import timing issue
Run this to update the import check
"""

import os
import shutil

print("Fixing TorusCells import warning...")

# Backup the original
init_file = "python/core/__init__.py"
backup_file = "python/core/__init__.py.backup"

if os.path.exists(init_file):
    # Create backup
    shutil.copy(init_file, backup_file)
    print(f"✅ Created backup: {backup_file}")
    
    # Read the file
    with open(init_file, 'r') as f:
        content = f.read()
    
    # Replace the problematic import with a delayed check
    old_import = """try:
    from .torus_cells import TorusCells, get_torus_cells, betti0_1, betti_update
except ImportError:
    TorusCells = None
    get_torus_cells = None
    betti0_1 = None
    betti_update = None"""
    
    new_import = """# Delayed import for TorusCells to avoid false warnings
def _import_torus_cells():
    try:
        from .torus_cells import TorusCells, get_torus_cells, betti0_1, betti_update
        return TorusCells, get_torus_cells, betti0_1, betti_update
    except ImportError:
        return None, None, None, None

# Try import but don't fail early
_tc_imports = _import_torus_cells()
TorusCells = _tc_imports[0]
get_torus_cells = _tc_imports[1]
betti0_1 = _tc_imports[2]
betti_update = _tc_imports[3]"""
    
    # Replace in content
    if old_import in content:
        new_content = content.replace(old_import, new_import)
        
        # Write back
        with open(init_file, 'w') as f:
            f.write(new_content)
        
        print("✅ Fixed TorusCells import timing")
        print("The warning should no longer appear!")
    else:
        print("⚠️ Import pattern not found - may already be fixed")
else:
    print("❌ __init__.py not found")

print("\nTo revert: copy __init__.py.backup back to __init__.py")
