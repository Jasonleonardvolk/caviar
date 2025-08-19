#!/usr/bin/env python3
"""
Run all tests for the ELFIN compiler.

This module discovers and runs all tests for the ELFIN compiler.
"""

import os
import sys
import unittest
from pathlib import Path

# Add the parent directory to the path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent  # compiler directory
sys.path.append(str(parent_dir))


def run_tests():
    """
    Discover and run all tests for the ELFIN compiler.
    
    Returns:
        True if all tests pass, False otherwise
    """
    # Discover all tests in this directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(str(script_dir), pattern="test_*.py")
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return True if all tests pass
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run the tests
    success = run_tests()
    
    # Return the appropriate exit code
    sys.exit(0 if success else 1)
