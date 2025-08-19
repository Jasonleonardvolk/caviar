#!/usr/bin/env python3
"""
Fix the exact duplicate code and extra brace issue
"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_brace_issue():
    """Remove duplicate code and fix the brace issue"""
    
    page_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\routes\+page.svelte")
    
    print("🔧 Fixing the duplicate code and brace issue...")
    
    with open(page_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the problematic section (around line 435-445)
    # We need to find where toggleHologramVideo ends and remove the duplicate
    
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for the end of toggleHologramVideo function
        if 'function toggleHologramVideo()' in line:
            # Add this line
            fixed_lines.append(line)
            i += 1
            
            # Keep adding lines until we find the proper closing brace
            brace_count = 1
            while i < len(lines) and brace_count > 0:
                line = lines[i]
                fixed_lines.append(line)
                
                # Count braces
                brace_count += line.count('{') - line.count('}')
                
                # If we just closed the function
                if brace_count == 0:
                    # Skip any duplicate code that follows
                    j = i + 1
                    while j < len(lines):
                        # Skip empty lines and duplicated localStorage code
                        if (lines[j].strip() == '' or 
                            'Save preference' in lines[j] or
                            'localStorage.setItem(\'tori-hologram-video\'' in lines[j] or
                            (lines[j].strip() == '}' and j < i + 10)):  # Skip extra braces near the function
                            j += 1
                        else:
                            break
                    
                    # Add separator comment
                    fixed_lines.append('\n')
                    fixed_lines.append('  // Subscribe to prosody stores\n')
                    
                    # Continue from where we stopped skipping
                    i = j
                    break
                else:
                    i += 1
        else:
            fixed_lines.append(line)
            i += 1
    
    # Write the fixed content
    with open(page_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("✅ Fixed the brace issue and removed duplicate code")
    print("✅ Added proper comment separator")

if __name__ == "__main__":
    fix_brace_issue()
