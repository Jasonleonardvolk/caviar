#!/usr/bin/env python3
"""
Test the unit kiwi feature for parameters in ELFIN.

This script tests the enhanced grammar that supports attaching units
to parameters (e.g., `x: velocity[m/s];`). It verifies that:
1. Files with unit specifications parse correctly
2. Legacy files without units remain valid
"""

import os
import sys
import re
from pathlib import Path

# Add the parent directory to the path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.append(str(parent_dir))

# Import the standalone parser
from standalone_parser import parse_file, parse


def test_legacy_system():
    """Test parsing a legacy system without units."""
    print("Testing legacy system parsing...")
    example_file = parent_dir / "examples" / "simple_system.elfin"
    
    if not example_file.exists():
        print(f"Error: Example file not found: {example_file}")
        return False
    
    try:
        ast = parse_file(example_file)
        print(f"✅ Successfully parsed legacy file with {len(ast.sections)} sections")
        return True
    except Exception as e:
        print(f"❌ Failed to parse legacy file: {e}")
        return False


def test_system_with_units():
    """Test parsing a system with unit specifications."""
    print("Testing system with units parsing...")
    example_file = parent_dir / "examples" / "system_with_units.elfin"
    
    if not example_file.exists():
        print(f"Error: Example file not found: {example_file}")
        return False
    
    try:
        # Using the standard parser won't extract units in our standalone parser,
        # but it should still parse the file correctly
        ast = parse_file(example_file)
        print(f"✅ Successfully parsed file with units, containing {len(ast.sections)} sections")
        
        # Extra verification: check that the file contains unit specifications
        with open(example_file, "r") as f:
            content = f.read()
        
        unit_pattern = r'\[\s*[a-zA-Z0-9\/\*\^\-\s\.]+\s*\]'
        units = re.findall(unit_pattern, content)
        
        if units:
            print(f"✅ Found {len(units)} unit specifications in the file:")
            for unit in units:
                print(f"  - {unit}")
        else:
            print("❌ No unit specifications found in the file")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Failed to parse file with units: {e}")
        return False


def main():
    """Run the tests."""
    print("=== Testing Unit Kiwi Feature ===\n")
    
    legacy_result = test_legacy_system()
    print()
    units_result = test_system_with_units()
    
    print("\n=== Test Results ===")
    print(f"Legacy system parsing: {'PASS' if legacy_result else 'FAIL'}")
    print(f"System with units parsing: {'PASS' if units_result else 'FAIL'}")
    
    # Return 0 if all tests pass, 1 otherwise
    return 0 if legacy_result and units_result else 1


if __name__ == "__main__":
    sys.exit(main())
