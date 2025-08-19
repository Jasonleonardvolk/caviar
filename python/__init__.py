"""
TORI/Saigon System
==================
Self-improving AI with continuous learning through
dynamic adapters, concept mesh, and holographic visualization.
"""

__version__ = "5.0.0"
__author__ = "TORI Team"
__license__ = "Proprietary"

# Core modules
from . import core
from . import training

# Optional imports with availability flags
try:
    from . import hott_integration
    HOTT_AVAILABLE = True
except ImportError:
    HOTT_AVAILABLE = False

# System information
def get_system_info():
    """Get system configuration and status."""
    import torch
    
    return {
        "version": __version__,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "hott_available": HOTT_AVAILABLE
    }

# Quick access to main components
from .core import SaigonInference, ConversationManager
from .training import train_lora_adapter, generate_synthetic_data

__all__ = [
    "core",
    "training",
    "get_system_info",
    "SaigonInference",
    "ConversationManager",
    "train_lora_adapter",
    "generate_synthetic_data"
]
