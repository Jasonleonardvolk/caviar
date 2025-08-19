# Unified Penrose Import Adapter
# This module provides a single canonical import path for the Penrose engine
# with transparent fallback to the pure-Python implementation

import logging

logger = logging.getLogger(__name__)

# Import ConceptMesh from interface module
from .interface import ConceptMesh

# Try to import the Rust engine first, fall back to Python if not available
try:
    import penrose_engine_rs as penrose
    PENROSE_BACKEND = "rust"
    logger.info("✅ Penrose engine initialized (Rust, SIMD)")
except ImportError:
    try:
        # Fall back to the pure Python implementation
        from penrose_projector import engine_numba as penrose
        PENROSE_BACKEND = "python-numba"
        logger.info("⚠️ Penrose engine initialized (Python, Numba JIT)")
    except ImportError:
        # Final fallback - import the basic engine
        from penrose_projector import engine as penrose
        PENROSE_BACKEND = "python"
        logger.warning("⚠️ Penrose engine initialized (Python, no acceleration)")

# Export the unified interface
__all__ = ['penrose', 'PENROSE_BACKEND', 'ConceptMesh']
