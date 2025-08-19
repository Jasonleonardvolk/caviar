#!/usr/bin/env python3
"""
Test the ELFIN parser.

This module tests the ELFIN parser and Megatron by parsing a simple ELFIN file
and verifying that the AST is constructed correctly.
"""

import os
import sys
import unittest
from pathlib import Path

# Add the parent directory to the path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent  # compiler directory
sys.path.append(str(parent_dir))

# Use relative imports
from grammar.parser import parse, parse_file
from ast import Node, Program, ImportStmt


class TestParser(unittest.TestCase):
    """Test the ELFIN parser and Megatron."""
    
    def test_parse_simple(self):
        """Test parsing a simple ELFIN program."""
        # Define a simple ELFIN program
        program = """
        // Simple test program
        import StdHelpers from "std/helpers.elfin";
        
        helpers TestHelpers {
            square(x) = x * x;
            cube(x) = x * x * x;
        }
        
        system TestSystem {
            continuous_state {
                x; y; theta;
            }
            
            input {
                v_x; v_y;
            }
            
            params {
                mass: 1.0;  // kg
                radius: 0.5;  // m
            }
            
            flow_dynamics {
                x_dot = v_x;
                y_dot = v_y;
                theta_dot = 0.1;  // constant rotation
            }
        }
        """
        
        # Parse the program
        ast = parse(program)
        
        # Check that the AST is a Program
        self.assertIsInstance(ast, Program)
        
        # Check that the program has one import
        self.assertEqual(len(ast.imports), 1)
        self.assertIsInstance(ast.imports[0], ImportStmt)
        self.assertEqual(ast.imports[0].section_name, "StdHelpers")
        self.assertEqual(ast.imports[0].file_path, "std/helpers.elfin")
        
        # Check that the program has sections
        self.assertGreaterEqual(len(ast.sections), 1)
    
    def test_parse_file(self):
        """Test parsing an ELFIN file."""
        # Create a temporary file with a simple ELFIN program
        test_file = script_dir / "test_file.elfin"
        with open(test_file, "w") as f:
            f.write("""
            // Simple test file
            system EmptySystem {
                continuous_state: [x, y];
                input: [u];
                params {
                    g: 9.81;
                }
                flow_dynamics {
                    x_dot = y;
                    y_dot = u - g;
                }
            }
            """)
        
        try:
            # Parse the file
            ast = parse_file(test_file)
            
            # Check that the AST is a Program
            self.assertIsInstance(ast, Program)
            
            # Check that the program has no imports
            self.assertEqual(len(ast.imports), 0)
            
            # Check that the program has one section
            self.assertEqual(len(ast.sections), 1)
        finally:
            # Clean up
            if test_file.exists():
                os.remove(test_file)


if __name__ == "__main__":
    unittest.main()
