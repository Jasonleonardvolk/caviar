"""
ELFIN language compiler and tools.

This package provides tools for working with the ELFIN control system
description language.
"""

# Create an import alias to make "from elfin.*" imports work
# This is needed because many existing files use this import pattern
import sys
import importlib

# Set up the alias if it doesn't exist yet
if 'elfin' not in sys.modules:
    sys.modules['elfin'] = sys.modules['alan_backend.elfin']
