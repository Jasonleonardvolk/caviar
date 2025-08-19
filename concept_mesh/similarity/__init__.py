"""
Penrose import shim.
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
* Tries the Rust extension first.
* Falls back to the existing Python/Numba projector.
"""

try:
    import penrose_engine_rs as penrose          # 🚀 Rust path
    BACKEND = "rust"
except ImportError:                              # 🐍 Python fallback
    from penrose_projector import engine as penrose
    BACKEND = "python-numba"

__all__ = ["penrose", "BACKEND"]
