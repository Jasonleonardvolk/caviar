"""
Test script for constant folding.

This script demonstrates the constant folding functionality by parsing an example
ELFIN file and running it through the compiler pipeline.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
root_dir = parent_dir.parent
sys.path.append(str(root_dir))

# Import the necessary modules
from elfin.standalone_parser import parse_file
from elfin.compiler.pipeline import CompilerPipeline

def main():
    """Test the constant folding functionality."""
    # Path to the test file
    test_file = os.path.join(parent_dir, 'examples', 'constant_folding_test.elfin')
    
    print(f"Parsing file: {test_file}")
    
    # Parse the file
    ast = parse_file(test_file)
    
    # Create a compiler pipeline
    pipeline = CompilerPipeline()
    
    # Process the AST
    print("Running compiler pipeline with constant folding...")
    pipeline.process(ast)
    
    # Get diagnostics
    diagnostics = pipeline.get_diagnostics()
    
    # Print diagnostics
    print(f"\nFound {len(diagnostics)} diagnostics:")
    for i, diag in enumerate(diagnostics, 1):
        print(f"{i}. {diag.severity.upper()}: {diag.message}")
    
    # Run again with folding disabled
    print("\nRunning compiler pipeline with constant folding disabled...")
    pipeline_no_fold = CompilerPipeline({'fold': False})
    pipeline_no_fold.process(ast)
    
    # Get diagnostics without folding
    no_fold_diagnostics = pipeline_no_fold.get_diagnostics()
    
    # Print diagnostics without folding
    print(f"\nFound {len(no_fold_diagnostics)} diagnostics without folding:")
    for i, diag in enumerate(no_fold_diagnostics, 1):
        print(f"{i}. {diag.severity.upper()}: {diag.message}")


if __name__ == "__main__":
    main()
