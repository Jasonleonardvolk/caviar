"""
ELFIN Stability Framework.

This package provides stability verification and enforcement capabilities
for the ELFIN DSL, including:

- Lyapunov function verification
- Neural Lyapunov learning
- Phase-synchronization monitoring
- Constraint IR generation
"""

from alan_backend.elfin.stability.constraint_ir import (
    ConstraintIR, VerificationResult, VerificationStatus, ConstraintType, ProofCache
)

from alan_backend.elfin.stability.learn_neural_lyap import (
    LyapunovNetwork, DynamicsModel, NeuralLyapunovLearner
)

# Import backends
import alan_backend.elfin.stability.backends

__all__ = [
    # Constraint IR
    'ConstraintIR', 'VerificationResult', 'VerificationStatus', 
    'ConstraintType', 'ProofCache',
    
    # Neural Lyapunov learning
    'LyapunovNetwork', 'DynamicsModel', 'NeuralLyapunovLearner',
    
    # Backend modules
    'backends'
]
