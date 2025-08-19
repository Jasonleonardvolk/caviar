"""Penrose Projector - High-performance similarity engine"""

# Import the basic engine (fallback)
from .engine import (
    PenroseEngine,
    compute_similarity,
    batch_similarity,
    is_available,
    penrose_engine
)

# Import the accelerated projector (the real deal!)
from .core import (
    PenroseProjector,
    project_sparse
)

__all__ = [
    # Basic engine exports (for compatibility)
    'PenroseEngine',
    'compute_similarity', 
    'batch_similarity',
    'is_available',
    'penrose_engine',
    
    # Accelerated projector exports (O(n^2.32) performance)
    'PenroseProjector',
    'project_sparse'
]
