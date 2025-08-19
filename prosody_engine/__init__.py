"""
Netflix-Killer Prosody Engine
=============================

A prosody engine so advanced it makes streaming services cry.
Integrates with TORI's holographic intelligence system.
"""

from .core import NetflixKillerProsodyEngine
from .api import prosody_router
from .streaming import ProsodyStreamProcessor
from .cultural import CulturalProsodyAdapter

__all__ = [
    'NetflixKillerProsodyEngine',
    'prosody_router',
    'ProsodyStreamProcessor',
    'CulturalProsodyAdapter'
]

__version__ = '1.0.0'