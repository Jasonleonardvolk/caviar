#!/usr/bin/env python3
"""
ELFIN Import Verification for CI

This script verifies that ELFIN files with imports can be processed correctly.
It's designed to be run in CI pipelines to ensure import functionality works.

- Processes files with imports
- Verifies the output matches expected semantics
- Returns non-zero exit code if verification fails

Usage:
  python verify_imports.py <directory_or_file>
"""

import os
import sys
import glob
from pathlib import Path

# Add the parent directory to the path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.append(str(parent_dir))

# Import the processor
from import_processor import process_imports

def verify_file(file_path):
    """
    Process a file with imports and verify the result.
    
    Args:
        file_path: Path to the ELFIN file to verify
        
    Returns:
        True if verification passed, False otherwise
    """
    print(f"Verifying imports in: {file_path}")
    
    # Create output path
    output_path = str(file_path) + '.processed'
    
    # Process imports
    processed_file = process_imports(file_path, output_path)
    
    if processed_file is None:
        print(f"❌ FAIL: Import processing failed for {file_path}")
        return False
    
    # Read the processed file
    with open(processed_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if the import statement is still present (should be replaced)
    if 'import ' in content and ' from "' in content:
        print(f"❌ FAIL: Import statement still present in {processed_file}")
        return False
    
    # Check if the helper functions are present (success criteria)
    expected_helpers = ['hAbs', 'hMin', 'hMax', 'wrapAngle']
    missing_helpers = [h for h in expected_helpers if h not in content]
    
    if missing_helpers:
        print(f"❌ FAIL: Missing expected helpers in {processed_file}: {', '.join(missing_helpers)}")
        return False
    
    print(f"✅ PASS: {file_path} imports verified successfully")
    return True

def verify_directory(directory):
    """
    Recursively verify all ELFIN files in a directory.
    
    Args:
        directory: Directory path to scan
        
    Returns:
        True if all verifications passed, False otherwise
    """
    elfin_files = glob.glob(os.path.join(directory, "**/*.elfin"), recursive=True)
    
    if not elfin_files:
        print(f"Warning: No ELFIN files found in {directory}")
        return True
    
    results = []
    for file_path in elfin_files:
        # Skip already processed files
        if file_path.endswith('.processed.elfin'):
            continue
            
        # Only process files with imports
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if 'import ' in content and ' from "' in content:
            results.append(verify_file(file_path))
    
    # Return overall success
    if not results:
        print("No files with imports found to verify.")
        return True
        
    success_count = sum(1 for r in results if r)
    total_count = len(results)
    
    print(f"\nVerification Summary: {success_count}/{total_count} files passed")
    return all(results)

def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("Usage: python verify_imports.py <directory_or_file>")
        return 1
    
    path = sys.argv[1]
    
    if os.path.isdir(path):
        success = verify_directory(path)
    else:
        success = verify_file(path)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
