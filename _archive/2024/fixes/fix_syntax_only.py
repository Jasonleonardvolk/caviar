#!/usr/bin/env python3
"""
Fix ONLY the syntax error - remove duplicate closing brace at line 445
"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_syntax_error_only():
    """Remove the duplicate closing brace at line 445"""
    
    page_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\routes\+page.svelte")
    
    print("🔧 Fixing syntax error in +page.svelte line 445...")
    
    # Read the file
    with open(page_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find and remove the duplicate closing brace
    # Based on the error, line 445 (index 444) has an extra }
    if len(lines) > 444 and lines[444].strip() == '}':
        print(f"Found duplicate brace at line 445: '{lines[444].strip()}'")
        
        # Check context - if the previous lines already closed the function
        if len(lines) > 443 and 'localStorage.setItem' in lines[442]:
            print("Confirmed: This is a duplicate closing brace after localStorage.setItem")
            # Remove the duplicate line
            lines.pop(444)
            print("✅ Removed duplicate closing brace")
        else:
            print("⚠️ Context doesn't match expected pattern, please check manually")
            return
    else:
        print("❌ Could not find the expected duplicate brace at line 445")
        print(f"Line 445 contains: '{lines[444].strip() if len(lines) > 444 else 'N/A'}'")
        return
    
    # Write the fixed content back
    with open(page_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("✅ Syntax error fixed!")
    print("\nYou can now restart the dev server.")

if __name__ == "__main__":
    fix_syntax_error_only()
