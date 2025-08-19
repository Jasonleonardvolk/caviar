#!/usr/bin/env python3
"""
ELFIN Circular Reference Checker

A simple entry point for checking ELFIN files for circular references.
This is designed to be used in CI pipelines or as a pre-commit hook.

Example usage:
  python check_circular_refs.py path/to/file.elfin
  python check_circular_refs.py path/to/directory/*.elfin
"""

import sys
import os
import glob
from pathlib import Path

# Add the analyzer directory to the path
script_dir = Path(__file__).parent
analyzer_dir = script_dir / "analyzer"
sys.path.append(str(analyzer_dir))

# Import the analyzer
from circular_analyzer import CircularReferenceAnalyzer

def check_files(file_patterns):
    """
    Check a list of files or file patterns for circular references.
    
    Args:
        file_patterns: List of file paths or glob patterns
        
    Returns:
        True if any issues were found, False otherwise
    """
    any_issues = False
    file_count = 0
    
    # Process all files
    for pattern in file_patterns:
        for file_path in glob.glob(pattern):
            if not file_path.endswith('.elfin'):
                continue
                
            file_count += 1
            print(f"Checking {file_path}...")
            
            # Analyze the file
            analyzer = CircularReferenceAnalyzer(file_path)
            analyzer.run_analysis()
            
            # Print results
            analyzer.print_summary()
            
            # Track if any issues were found
            if analyzer.has_circular_references():
                any_issues = True
    
    # Print summary
    print(f"\nFiles checked: {file_count}")
    if any_issues:
        print("❌ FAIL: Circular references detected!")
    else:
        print("✅ PASS: No circular references detected.")
    
    return any_issues

def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("Usage: python check_circular_refs.py <file_path> [file_path2 ...]")
        return 1
    
    file_patterns = sys.argv[1:]
    has_issues = check_files(file_patterns)
    
    # Return non-zero exit code if issues were found
    return 1 if has_issues else 0

if __name__ == "__main__":
    sys.exit(main())
