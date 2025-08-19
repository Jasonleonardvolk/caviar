#!/usr/bin/env python3
"""
Master TypeScript Error Fixer
Runs all fix scripts in the correct order
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name):
    """Run a Python script and return success status"""
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    project_root = Path(r"D:\Dev\kha")
    
    print("=" * 60)
    print("TypeScript Error Master Fixer")
    print("=" * 60)
    print()
    
    scripts = [
        ("fix_typescript_errors.py", "Applying automatic fixes for common errors..."),
        ("create_missing_stubs.py", "Creating missing module stub files..."),
        ("apply_quick_fixes.py", "Applying additional quick fixes...")
    ]
    
    all_success = True
    
    for script, description in scripts:
        script_path = project_root / script
        if not script_path.exists():
            print(f"Warning: {script} not found!")
            all_success = False
            continue
        
        print(f"\n{description}")
        print("-" * 40)
        
        if not run_script(str(script_path)):
            all_success = False
            print(f"Failed to run {script}")
        else:
            print(f"Successfully completed {script}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if all_success:
        print("All fixes applied successfully!")
    else:
        print("Some fixes failed - check the output above")
    
    print("\nNext Steps:")
    print("1. Install required packages:")
    print("   cd frontend")
    print("   npm install --save-dev @types/node")
    print("   npm install react @types/react svelte")
    print()
    print("2. Run TypeScript compiler to check remaining errors:")
    print("   npx tsc -p frontend/tsconfig.json --noEmit")
    print()
    print("3. Check the following files for manual review:")
    print("   - TYPESCRIPT_FIXES_MANUAL.md (detailed manual fixes)")
    print("   - typescript_fixes_report.json (automated fixes report)")
    print("   - PACKAGE_UPDATES.txt (required npm packages)")
    print()
    print("4. For any remaining import errors, verify that:")
    print("   - The file paths are correct for your project structure")
    print("   - The tsconfig.json paths are properly configured")
    print("   - All required dependencies are installed")

if __name__ == "__main__":
    main()
