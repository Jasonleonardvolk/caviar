"""
Physics package for soliton-based computation
Handles nonlinear wave dynamics and χ³ interactions
"""

from .soliton_engine import SolitonEngine, create_engine

# Export default parameters for easy access
from .soliton_engine import (
    DEFAULT_GAMMA_EFF,
    DEFAULT_RING_BOOST_DB,
    DEFAULT_SNR_THRESHOLD,
    DEFAULT_DUTY_CYCLE
)

__all__ = [
    'SolitonEngine',
    'create_engine',
    'DEFAULT_GAMMA_EFF',
    'DEFAULT_RING_BOOST_DB', 
    'DEFAULT_SNR_THRESHOLD',
    'DEFAULT_DUTY_CYCLE'
]
