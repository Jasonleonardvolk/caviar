"""
Penrose Engine Alias Module
Bridges penrose_engine_rs to penrose_engine for adapter compatibility
"""

# Import everything from the actual Rust backend
from penrose_engine_rs import *

# Re-export the module's documentation and attributes
import penrose_engine_rs
__doc__ = penrose_engine_rs.__doc__
if hasattr(penrose_engine_rs, "__all__"):
    __all__ = penrose_engine_rs.__all__
