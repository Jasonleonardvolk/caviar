#!/usr/bin/env python3
"""
FINAL GREMLIN SLAYER - Fix the last remaining syntax error
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    print("\n" + "🎯 "*20)
    print("FINAL GREMLIN SLAYER")
    print("🎯 "*20 + "\n")

def fix_syntax_error():
    """Fix the unmatched parenthesis in prajna_api.py"""
    print("🔧 Fixing syntax error in prajna_api.py...")
    
    prajna_file = Path("prajna_api.py")
    if not prajna_file.exists():
        print("❌ prajna_api.py not found!")
        return False
    
    # Read the file
    with open(prajna_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find and fix line 595 (index 594)
    if len(lines) > 594:
        line_595 = lines[594]
        if line_595.strip() == ')':
            print(f"   Found extra parenthesis on line 595: '{line_595.strip()}'")
            # Remove this line
            lines.pop(594)
            
            # Write back
            with open(prajna_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            print("✅ Removed extra closing parenthesis")
            return True
        else:
            # Look for the pattern in nearby lines
            for i in range(max(0, 590), min(len(lines), 600)):
                if 'return {' in lines[i] and lines[i].rstrip().endswith('})'):
                    print(f"   Found issue on line {i+1}: extra ) at end")
                    lines[i] = lines[i].rstrip()[:-1] + '\n'
                    
                    with open(prajna_file, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    
                    print("✅ Fixed syntax error")
                    return True
    
    print("⚠️ Could not automatically fix - manual intervention needed")
    return False

def verify_syntax():
    """Verify Python syntax after fix"""
    print("\n🔍 Verifying Python syntax...")
    
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", "prajna_api.py"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Python syntax is now valid!")
        return True
    else:
        print("❌ Syntax error still present:")
        print(result.stderr)
        return False

def main():
    print_banner()
    
    # Fix the syntax error
    if fix_syntax_error():
        # Verify the fix
        if verify_syntax():
            print("\n🎉 SUCCESS! The final gremlin has been slain!")
            print("\n🚀 You can now run:")
            print("   poetry run python enhanced_launcher.py")
            print("\n✨ TORI should launch successfully!")
        else:
            print("\n⚠️ Fix was applied but syntax is still invalid")
            print("Please check prajna_api.py manually around line 595")
    else:
        print("\n⚠️ Could not apply automatic fix")
        print("\nManual fix required:")
        print("1. Open prajna_api.py")
        print("2. Go to line 595")
        print("3. Remove the extra closing parenthesis ')' ")
        print("4. Save the file")

if __name__ == "__main__":
    main()
