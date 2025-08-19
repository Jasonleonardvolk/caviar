from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# {PROJECT_ROOT}\cog\__init__.py

"""
TORI Cognitive Framework
========================

A comprehensive metacognitive architecture implementing:
- MetaCognitive Manifolds with Riemannian geometry
- Reflective operators with natural gradients
- Self-modification via free energy minimization
- Curiosity-driven exploration
- Transfer learning via persistent homology
- Cognitive dynamics with stochastic evolution
- Fixed point computation
- Consciousness monitoring with IIT
- Metacognitive tower with ∞-categorical structure
- Knowledge sheaf for distributed cognition
"""

__version__ = "1.0.0"
__author__ = "TORI Framework Team"

from .manifold import MetaCognitiveManifold
from .reflective import ReflectiveOperator
from .self_modification import SelfModificationOperator, compute_free_energy
from .curiosity import CuriosityFunctional
from .transfer import TransferMorphism
from .dynamics import CognitiveDynamics
from .fixed_point import find_fixed_point
from .utils import set_random_seed, numeric_gradient, compute_iit_phi
from .monitoring import ConsciousnessMonitor, LyapunovStabilizer
from .architecture import MetacognitiveTower, KnowledgeSheaf
from .connectivity import create_connectivity_matrix

__all__ = [
    'MetaCognitiveManifold',
    'ReflectiveOperator',
    'SelfModificationOperator',
    'compute_free_energy',
    'CuriosityFunctional',
    'TransferMorphism',
    'CognitiveDynamics',
    'find_fixed_point',
    'set_random_seed',
    'numeric_gradient',
    'compute_iit_phi',
    'ConsciousnessMonitor',
    'LyapunovStabilizer',
    'MetacognitiveTower',
    'KnowledgeSheaf',
    'create_connectivity_matrix'
]