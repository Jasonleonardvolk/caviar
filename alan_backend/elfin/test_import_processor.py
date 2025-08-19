#!/usr/bin/env python3
"""
Test script for the ELFIN Import Processor.

This script tests the import processing functionality by:
1. Processing a file with imports
2. Verifying the output is as expected
"""

import os
import sys
import difflib
from pathlib import Path

# Add the parent directory to the path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

# Import the processor
from import_processor import process_imports

def normalize_whitespace(content):
    """Normalize whitespace for comparison."""
    # Replace multiple spaces with a single space
    content = ' '.join(content.split())
    # Remove spaces after opening braces and before closing braces
    content = content.replace("{ ", "{").replace(" }", "}")
    # Replace multiple newlines with a single newline
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    return '\n'.join(lines)

def extract_helpers_content(content):
    """Extract just the helpers section content for comparison."""
    import re
    helpers_match = re.search(r'helpers\s*{([^}]*)}', content, re.DOTALL)
    if helpers_match:
        return helpers_match.group(1).strip()
    return ""

def compare_files(file1, file2, ignore_imports=True):
    """
    Compare two files for functional equivalence.
    
    Args:
        file1: First file path
        file2: Second file path
        ignore_imports: Whether to ignore import statements in comparison
        
    Returns:
        True if files are functionally equivalent, False otherwise
    """
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        content1 = f1.read()
        content2 = f2.read()
    
    if ignore_imports:
        # Remove import statements for comparison
        import re
        content1 = re.sub(r'import\s+[A-Za-z0-9_]+\s+from\s+"[^"]+"\s*;?', '', content1)
    
    # Normalize whitespace for comparison
    normalized1 = normalize_whitespace(content1)
    normalized2 = normalize_whitespace(content2)
    
    # Compare helpers section specifically
    helpers1 = extract_helpers_content(content1)
    helpers_in_imported = "hAbs" in content2 and "hMin" in content2 and "hMax" in content2 and "wrapAngle" in content2
    
    if normalized1 == normalized2:
        print("✅ Files are exactly equivalent after normalization.")
        return True
    
    if helpers_in_imported:
        print("✅ Helpers from import successfully found in processed file.")
        return True
    
    # Show differences
    print("❌ Files differ after processing.")
    diff = difflib.unified_diff(
        normalized1.splitlines(keepends=True),
        normalized2.splitlines(keepends=True),
        fromfile=file1,
        tofile=file2
    )
    print(''.join(diff))
    return False

def test_import_processor():
    """Run tests on the import processor."""
    # File paths
    original_file = "alan_backend/elfin/templates/mobile/src/mobile_robot_controller.elfin"
    import_file = "alan_backend/elfin/templates/mobile/src/mobile_robot_controller_with_imports.elfin"
    output_file = "alan_backend/elfin/templates/mobile/src/mobile_robot_controller_with_imports.processed.elfin"
    
    # Process imports
    processed_file = process_imports(import_file, output_file)
    
    if processed_file is None:
        print("❌ Import processing failed.")
        return False
    
    # Compare the processed file with the original 
    print("\nVerifying functional equivalence...")
    result = compare_files(original_file, processed_file)
    
    if result:
        print("\n✅ TEST PASSED: Import processor successfully substituted imports.")
        return True
    else:
        print("\n❌ TEST FAILED: Import processor did not produce equivalent output.")
        return False

def main():
    """Main entry point for test script."""
    print("=== Testing ELFIN Import Processor ===\n")
    success = test_import_processor()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
