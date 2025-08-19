#!/usr/bin/env python3
"""
Fix the exact duplicate code issue found at lines 441-445
"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_duplicate_code():
    """Remove the duplicate code that's outside the function"""
    
    page_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\routes\+page.svelte")
    
    assert page_path.exists(), "+page.svelte not found; aborting"
    
    print("🔧 Fixing duplicate code in +page.svelte...")
    
    with open(page_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Remove lines 441-445 which contain:
    # // Subscribe to prosody stores
    # // Subscribe to prosody stores
    #   if (browser) {
    #     localStorage.setItem('tori-hologram-video', String(hologramVisualizationEnabled));
    #   }
    
    fixed_lines = []
    skip_lines = False
    skip_count = 0
    
    for i, line in enumerate(lines):
        # Check if we're at the duplicate section (line 441)
        if i == 440 and i + 5 < len(lines):
            # Verify this is the duplicate section
            if (lines[i].strip() == "// Subscribe to prosody stores" and
                lines[i+1].strip() == "// Subscribe to prosody stores" and
                "if (browser) {" in lines[i+2] and
                "localStorage.setItem('tori-hologram-video'" in lines[i+3] and
                lines[i+4].strip() == "}"):
                
                print(f"✅ Found duplicate code at lines {i+1}-{i+5}")
                print("   Removing:")
                for j in range(5):
                    print(f"   Line {i+j+1}: {lines[i+j].rstrip()}")
                
                skip_lines = True
                skip_count = 5
                continue
        
        # Skip the duplicate lines
        if skip_lines and skip_count > 0:
            skip_count -= 1
            if skip_count == 0:
                skip_lines = False
            continue
        
        fixed_lines.append(line)
    
    # Write the fixed content
    with open(page_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("\n✅ Fixed! Removed duplicate code block")
    print("   The syntax error should be resolved now")

if __name__ == "__main__":
    fix_duplicate_code()
