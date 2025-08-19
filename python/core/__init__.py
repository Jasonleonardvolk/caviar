"""
Core import surface for TORI hybrid holography.

This module exposes stable, production-ready imports so application code
(e.g., hybrid_holographic_core.py, FastAPI endpoints, jobs) doesn't need to
know file-level locations. Treat these names as your public API.
"""

from .phase_to_depth import phase_to_depth, load_phase_data
from .mesh_exporter import MeshExporter, MeshUpdateWatcher
from .adapter_loader import AdapterLoader

# Optional: expose other core pieces already in your tree if you want one-stop imports
try:
    from .phase_encode import encode_phase_map  # if present in python\core\phase_encode.py
except Exception:
    encode_phase_map = None  # keep surface stable if not available yet

try:
    from .concept_mesh import ConceptMesh
except Exception:
    ConceptMesh = None

try:
    from .memory_vault import MemoryVault
except Exception:
    MemoryVault = None

__all__ = [
    "phase_to_depth",
    "load_phase_data",
    "MeshExporter",
    "MeshUpdateWatcher",
    "AdapterLoader",
    "encode_phase_map",
    "ConceptMesh",
    "MemoryVault",
]