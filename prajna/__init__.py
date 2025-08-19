"""
Prajna Package Initialization
============================

This module initializes the Prajna package and provides easy imports
for all core components.
"""

# Import core components (not API to avoid circular imports)
from .core.prajna_mouth import PrajnaLanguageModel, generate_prajna_response
from .memory.context_builder import build_context, ContextResult
from .memory.soliton_interface import SolitonMemoryInterface
from .memory.concept_mesh_api import ConceptMeshAPI
from .audit.alien_overlay import audit_prajna_answer, ghost_feedback_analysis
from .config.prajna_config import PrajnaConfig, load_config

__version__ = "1.0.0"
__author__ = "TORI Development Team"
__description__ = "Prajna: TORI's Voice and Language Model"

# Easy imports for external use
__all__ = [
    # Language Model
    "PrajnaLanguageModel",
    "generate_prajna_response",
    
    # Memory Systems
    "build_context",
    "ContextResult",
    "SolitonMemoryInterface", 
    "ConceptMeshAPI",
    
    # Audit Systems
    "audit_prajna_answer",
    "ghost_feedback_analysis",
    
    # Configuration
    "PrajnaConfig",
    "load_config",
]

def get_version():
    """Get Prajna version"""
    return __version__

def get_info():
    """Get Prajna system information"""
    return {
        "name": "Prajna",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "voice_of": "TORI Cognitive System"
    }
