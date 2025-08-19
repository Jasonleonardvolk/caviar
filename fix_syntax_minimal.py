#!/usr/bin/env python3
"""
Minimal fix for the syntax error ONLY
Other changes should be done properly through Svelte stores and components
"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_syntax_error_only():
    """Fix ONLY the duplicate closing brace syntax error"""
    
    page_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\routes\+page.svelte")
    
    assert page_path.exists(), "+page.svelte not found; aborting"
    
    print("🔧 Fixing syntax error in +page.svelte...")
    
    # Read the file
    with open(page_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the duplicate closing brace around line 445
    # Look for pattern:
    # localStorage.setItem('tori-hologram-video', String(hologramVisualizationEnabled));
    # }
    # }  <-- This is the extra one
    
    fixed_lines = []
    i = 0
    removed = False
    
    while i < len(lines):
        line = lines[i]
        
        # Look for the localStorage.setItem line
        if "localStorage.setItem('tori-hologram-video'" in line and i + 2 < len(lines):
            # Check if next two lines are both closing braces
            if lines[i + 1].strip() == '}' and lines[i + 2].strip() == '}':
                # Add the localStorage line and first closing brace
                fixed_lines.append(line)
                fixed_lines.append(lines[i + 1])
                # Skip the duplicate closing brace
                i += 3  # Skip to line after the duplicate
                removed = True
                print(f"✅ Removed duplicate closing brace at line {i + 2}")
                continue
        
        fixed_lines.append(line)
        i += 1
    
    if removed:
        # Write the fixed content
        with open(page_path, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)
        print("✅ Syntax error fixed!")
    else:
        print("⚠️ Could not find the duplicate closing brace pattern")
        print("   Please check the file manually")

if __name__ == "__main__":
    fix_syntax_error_only()
