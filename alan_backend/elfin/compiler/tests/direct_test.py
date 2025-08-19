#!/usr/bin/env python3
"""
Direct tests for the ELFIN compiler.

These tests use direct imports rather than package imports,
which are more robust for testing in development environments.
"""

import unittest
import sys
import os
from pathlib import Path
import importlib.util

# Get absolute paths to modules we want to import
current_dir = Path(__file__).parent.absolute()
grammar_dir = current_dir.parent / "grammar"
ast_dir = current_dir.parent / "ast"

# Import parser module directly by path
parser_path = grammar_dir / "parser.py"
spec = importlib.util.spec_from_file_location("parser", parser_path)
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)

# Import nodes module directly by path
nodes_path = ast_dir / "nodes.py"
spec = importlib.util.spec_from_file_location("nodes", nodes_path)
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)

# Import megatron module directly by path
megatron_path = ast_dir / "megatron.py"
spec = importlib.util.spec_from_file_location("megatron", megatron_path)
megatron = importlib.util.module_from_spec(spec)
spec.loader.exec_module(megatron)

# Extracted names for convenience
parse = parser.parse
parse_file = parser.parse_file
ELFINSyntaxError = parser.ELFINSyntaxError
ELFINMegatron = megatron.ELFINMegatron
Node = nodes.Node
Program = nodes.Program
ImportStmt = nodes.ImportStmt


class TestDirectImports(unittest.TestCase):
    """Tests that use direct imports."""

    def test_basic_parsing(self):
        """Test basic parsing functionality."""
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
        
        # Parse the program
        try:
            ast = parse(simple_program)
            self.assertIsInstance(ast, Program)
            self.assertEqual(len(ast.imports), 0)
            self.assertEqual(len(ast.sections), 1)
            print("Basic parsing test passed!")
        except Exception as e:
            self.fail(f"Parsing failed: {e}")


if __name__ == "__main__":
    unittest.main()
