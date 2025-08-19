#!/usr/bin/env python3
"""
Remove the duplicate code block that's outside the function
"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_duplicate_code():
    """Remove the duplicate localStorage code that's outside the function"""
    
    page_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\routes\+page.svelte")
    
    print("🔧 Removing duplicate code block...")
    
    with open(page_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and remove the duplicate block
    # The pattern is: end of toggleHologramVideo function, then duplicate code
    import re
    
    # This pattern matches the end of toggleHologramVideo and the duplicate block
    pattern = r'(function toggleHologramVideo\(\) {[^}]+}\s*}\s*)\n\s*// Save preference\s*\n\s*if \(browser\) {\s*\n\s*localStorage\.setItem\(\'tori-hologram-video\', String\(hologramVisualizationEnabled\)\);\s*\n\s*}\s*\n\s*}'
    
    # Replace with just the function (without the duplicate)
    content = re.sub(pattern, r'\1', content, flags=re.DOTALL)
    
    with open(page_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Removed duplicate code block")
    print("\nThe syntax error should be fixed now!")

if __name__ == "__main__":
    fix_duplicate_code()
