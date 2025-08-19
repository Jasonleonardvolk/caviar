"""
Import fixes for concept_mesh module
Add this at the top of any file with relative imports
"""

import sys
from pathlib import Path

# Fix for relative imports when running as script
if __package__ is None and "." not in __name__:
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    
# Now relative imports will work:
# from .loader import load_mesh
# from .similarity import penrose
