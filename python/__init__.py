"""
Lightweight package init for TORI.

Deliberately avoids importing heavy subpackages (e.g., training, torch)
at import time so that `import python.core.fractal_soliton_memory` does
not pull GPU stacks during test collection.
"""
__all__ = []  # explicit exports come from submodules
