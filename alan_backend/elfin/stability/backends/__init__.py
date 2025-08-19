"""
ELFIN stability verification backends.

This package contains various backend implementations for Lyapunov
stability verification, including:

- SOS: Sum-of-Squares programming for polynomial Lyapunov functions
- MILP: Mixed-Integer Linear Programming for neural network Lyapunov functions
- SMT: Satisfiability Modulo Theories for general verification

Each backend provides a uniform interface for verifying stability properties
using the ConstraintIR representation.
"""

from alan_backend.elfin.stability.backends.sos_backend import SOSToolsBackend, SOSVerifier

__all__ = ['SOSToolsBackend', 'SOSVerifier']
