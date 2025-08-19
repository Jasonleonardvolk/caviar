#!/usr/bin/env python3
"""
Post-merge sanity checker for the ELFIN Stability Framework.

This script runs a series of validation checks to ensure the framework
components are working correctly together after integration.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Add the root directory to Python path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import standalone test script
try:
    from standalone_sanity_checks import main
    
    if __name__ == "__main__":
        sys.exit(main())
        
except ImportError as e:
    logging.error(f"Failed to import standalone_sanity_checks: {e}")
    logging.error("Make sure standalone_sanity_checks.py exists in the project root.")
    
    if __name__ == "__main__":
        sys.exit(1)
