"""
Barrier certificate module for ELFIN.

This module provides functionality for learning, verifying, and using barrier
certificates for safety verification. Barrier certificates provide formal
guarantees that a system will not enter unsafe regions of the state space.
"""

# Export key classes and functions for IDE autocompletion and public API

# Core barrier function and learning
from alan_backend.elfin.barrier.learner import BarrierFunction, BarrierLearner

# Verification
from alan_backend.elfin.barrier.sos_verifier import (
    SOSVerifier, 
    VerificationResult
)

# High-level agent
from alan_backend.elfin.barrier.barrier_bridge_agent import (
    BarrierBridgeAgent,
    create_double_integrator_agent
)

# Formal verification with SOS
try:
    from alan_backend.elfin.barrier.sos_mosek import (
        MosekSOSVerifier, 
        SOSProofCertificate,
        verify_with_mosek
    )
except ImportError:
    # Mosek not available
    pass

# Direct CLI commands
from alan_backend.elfin.barrier.cli import add_barrier_commands

# Version
__version__ = "0.3.0"

# Public API
__all__ = [
    # Core classes
    "BarrierFunction",
    "BarrierLearner",
    "BarrierBridgeAgent",
    
    # Verification
    "SOSVerifier",
    "VerificationResult",
    "MosekSOSVerifier",
    "SOSProofCertificate",
    "verify_with_mosek",
    
    # Helper functions
    "create_double_integrator_agent",
    "add_barrier_commands",
]
