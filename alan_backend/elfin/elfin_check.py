#!/usr/bin/env python3
"""
ELFIN Check - Command Line Tool

A command-line tool for checking ELFIN files for circular references and other issues.
This is the main entry point for the `elfin check` command.
"""

import sys
import os
from pathlib import Path

# Ensure the analyzer module is in the path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir.parent.parent))

# Import the CLI module
from alan_backend.elfin.analyzer.cli import main

if __name__ == "__main__":
    # Simply delegate to the CLI main function
    main()
