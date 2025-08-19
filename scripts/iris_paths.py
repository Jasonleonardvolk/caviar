# scripts/iris_paths.py
"""
Runtime path resolver for Python scripts
Resolves ${IRIS_ROOT} tokens and {PROJECT_ROOT} to actual filesystem paths
"""
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def root() -> Path:
    """Get the project root directory from env or relative to this file"""
    if "IRIS_ROOT" in os.environ:
        return Path(os.environ["IRIS_ROOT"])
    # Go up from scripts/iris_paths.py to project root
    return Path(__file__).resolve().parents[1]

def resolve(*parts: str) -> Path:
    """Resolve path parts relative to project root"""
    return root().joinpath(*parts)

def replace_tokens(s: str) -> str:
    """Replace ${IRIS_ROOT} tokens with actual path"""
    return s.replace("${IRIS_ROOT}", str(root()))

# For compatibility with refactored Python files that use {PROJECT_ROOT}
PROJECT_ROOT = root()

# Usage examples:
# from scripts.iris_paths import resolve, PROJECT_ROOT
# psi_morphon = resolve("hott_integration", "psi_morphon.py")
# config_file = PROJECT_ROOT / "config" / "settings.json"
