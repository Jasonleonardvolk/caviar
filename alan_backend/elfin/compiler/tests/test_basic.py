#!/usr/bin/env python3
"""
Basic tests for the ELFIN compiler.

These tests don't rely on complex imports and provide a simple verification
that the basic components are working.
"""

import unittest
import sys
import os
from pathlib import Path

# Make sure we can import modules directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# We import Lark directly from the lark package
from lark import Lark


class TestBasic(unittest.TestCase):
    """Basic tests for ELFIN components."""

    def test_imports(self):
        """Test that all required modules can be imported."""
        # Test importing Lark
        import lark
        self.assertTrue(hasattr(lark, 'Lark'))
        
        # Test importing our own modules
        from alan_backend.elfin.compiler.grammar import parser
        self.assertTrue(hasattr(parser, 'parse'))
        
        from alan_backend.elfin.compiler.ast import megatron
        self.assertTrue(hasattr(megatron, 'ELFINMegatron'))
        
        # Test importing nodes
        from alan_backend.elfin.compiler.ast import nodes
        self.assertTrue(hasattr(nodes, 'Node'))
        self.assertTrue(hasattr(nodes, 'Program'))
        
        # Test importing visitors
        from alan_backend.elfin.compiler.ast import visitor
        self.assertTrue(hasattr(visitor, 'NodeVisitor'))
        
        # Import main compiler
        from alan_backend.elfin.compiler import main
        self.assertTrue(hasattr(main, 'compile_file'))
        
        print("All imports successful!")


if __name__ == "__main__":
    unittest.main()
