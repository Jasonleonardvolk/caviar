#!/usr/bin/env python
"""
Standalone Demo of the ELFIN Unit Annotation System

This script demonstrates how to use the ELFIN Unit Annotation System without
relying on the main ELFIN parser. It uses direct imports from our new modules.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Direct imports from our new modules (avoiding the parser dependency)
from alan_backend.elfin.units.checker import check_elfin_file, DimensionError
from alan_backend.elfin.codegen.rust import generate_rust_code


def main():
    """Main entry point for the demo."""
    print("ELFIN Unit Annotation System Demo (Standalone)")
    print("=============================================")
    
    # Get path to example ELFIN file
    elfin_file = os.path.join(os.path.dirname(__file__), "pendulum_units.elfin")
    
    # Check dimensional consistency
    print("\n1. Checking dimensional consistency...")
    try:
        errors = check_elfin_file(elfin_file)
        
        if errors:
            print("Found dimensional errors:")
            for expr, error in errors:
                print(f"  {expr}: {error}")
        else:
            print("✅ No dimensional errors found!")
    except Exception as e:
        print(f"Error checking file: {e}")
        return
    
    # Generate Rust code with unit safety
    print("\n2. Generating Rust code with dimensional safety...")
    output_dir = os.path.join(os.path.dirname(__file__), "generated")
    try:
        output_file = generate_rust_code(elfin_file, output_dir, use_units=True)
        print(f"✅ Generated Rust code: {output_file}")
        
        # Print part of the generated file
        with open(output_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            print("\nSnippet from the generated Rust code (with units):")
            for line in lines[:20]:  # Print first 20 lines
                print(f"  {line}")
            print("  ...")
    except Exception as e:
        print(f"Error generating code: {e}")
        return
    
    # Generate Rust code without unit safety (for embedded targets)
    print("\n3. Generating Rust code without dimensional safety (for embedded targets)...")
    output_dir_no_units = os.path.join(os.path.dirname(__file__), "generated_no_units")
    try:
        output_file = generate_rust_code(elfin_file, output_dir_no_units, use_units=False)
        print(f"✅ Generated Rust code: {output_file}")
        
        # Print part of the generated file
        with open(output_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            print("\nSnippet from the generated Rust code (without units):")
            for line in lines[:20]:  # Print first 20 lines
                print(f"  {line}")
            print("  ...")
    except Exception as e:
        print(f"Error generating code: {e}")
        return
    
    # Success
    print("\n✅ Demo completed successfully!")
    print(f"\nGenerated files:")
    print(f"  - {output_dir}/pendulum.rs (with dimensional safety)")
    print(f"  - {output_dir_no_units}/pendulum.rs (without dimensional safety)")


if __name__ == "__main__":
    main()
