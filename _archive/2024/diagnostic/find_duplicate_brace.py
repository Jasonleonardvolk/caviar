#!/usr/bin/env python3
"""
Find and show the exact duplicate brace location
"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def find_duplicate_brace():
    """Find the exact location of the duplicate brace"""
    
    page_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\routes\+page.svelte")
    
    with open(page_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Total lines in file: {len(lines)}")
    print("\nLooking for duplicate pattern around toggleHologramVideo...")
    
    # Find the toggleHologramVideo function
    for i, line in enumerate(lines):
        if 'function toggleHologramVideo()' in line:
            print(f"\nFound toggleHologramVideo at line {i + 1}")
            # Show context
            start = max(0, i - 5)
            end = min(len(lines), i + 30)
            
            print("\nContext:")
            for j in range(start, end):
                marker = ">>>" if j == i else "   "
                print(f"{j+1:4d} {marker} {lines[j].rstrip()}")
                
                # Look for duplicate patterns
                if j > i and "localStorage.setItem('tori-hologram-video'" in lines[j]:
                    print(f"\n!!! Found localStorage.setItem at line {j + 1}")
                    # Check for duplicate closing braces after this
                    for k in range(j + 1, min(j + 5, len(lines))):
                        print(f"{k+1:4d}     {lines[k].rstrip()}")

if __name__ == "__main__":
    find_duplicate_brace()
