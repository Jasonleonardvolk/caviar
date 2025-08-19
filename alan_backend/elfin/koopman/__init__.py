"""
Koopman operator analysis and Lyapunov function generation.

This module provides tools for analyzing dynamical systems through
Koopman operator theory, including Extended Dynamic Mode Decomposition
(EDMD) and spectral Lyapunov function generation.
"""

from .dictionaries import (
    fourier_dict, 
    rbf_dict, 
    poly_dict, 
    create_dictionary,
    StandardDictionary
)
from .edmd import (
    edmd_fit, 
    kfold_validation,
    estimate_optimal_dict_size
)
from .koopman_lyap import (
    KoopmanLyapunov,
    create_koopman_lyapunov,
    get_stable_modes
)

__all__ = [
    # Dictionary functions
    'fourier_dict',
    'rbf_dict',
    'poly_dict',
    'create_dictionary',
    'StandardDictionary',
    
    # EDMD functions
    'edmd_fit',
    'kfold_validation',
    'estimate_optimal_dict_size',
    
    # Lyapunov generation
    'KoopmanLyapunov',
    'create_koopman_lyapunov',
    'get_stable_modes',
    
    # Bridge agent
    'KoopmanBridgeAgent',
    'create_pendulum_agent',
    'create_vdp_agent'
]

# Import bridge agent
from .koopman_bridge_agent import (
    KoopmanBridgeAgent,
    create_pendulum_agent,
    create_vdp_agent
)
