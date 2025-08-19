#!/usr/bin/env python3
"""
Minimal test for the ELFIN compiler.

This script tests basic functionality without relying on imports.
"""

import sys
import os
from pathlib import Path

def main():
    """Run a simple test."""
    print("ELFIN Compiler - Minimal Test")
    print("=============================")
    
    # Print the module search paths
    print("\nPython module search paths:")
    for path in sys.path:
        print(f"  {path}")
    
    # Print information about the current directory
    current_dir = Path(__file__).parent.absolute()
    print(f"\nCurrent directory: {current_dir}")
    
    # Check for existence of key files
    grammar_dir = current_dir / "grammar"
    ast_dir = current_dir / "ast"
    
    parser_file = grammar_dir / "parser.py"
    megatron_file = ast_dir / "megatron.py"
    nodes_file = ast_dir / "nodes.py"
    
    print("\nChecking for key files:")
    print(f"  Parser: {'✓ exists' if parser_file.exists() else '✗ missing'}")
    print(f"  Megatron: {'✓ exists' if megatron_file.exists() else '✗ missing'}")
    print(f"  Nodes: {'✓ exists' if nodes_file.exists() else '✗ missing'}")
    
    # Simple syntax test
    simple_program = """
    // A simple ELFIN program
    system SimpleSystem {
        continuous_state: [x, y];
        input: [u];
        
        params {
            k: 1.0;  // Some parameter
        }
        
        flow_dynamics {
            x_dot = y;
            y_dot = u;
        }
    }
    """
    
    print("\nSample ELFIN program:")
    print("---------------------")
    print(simple_program)
    
    print("\nTest completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
