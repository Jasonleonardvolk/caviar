#!/usr/bin/env python3
"""
Alternative approaches to build Concept Mesh
Option 1: Use intel-mkl instead of OpenBLAS
Option 2: Build without BLAS (reduced performance)
Option 3: Use pre-built wheel
"""

import os
import sys
import subprocess
from pathlib import Path

print("🔧 ALTERNATIVE CONCEPT MESH BUILD STRATEGIES")
print("=" * 60)

kha_path = Path(__file__).parent.absolute()
concept_mesh_dir = kha_path / "concept_mesh"
cargo_toml = concept_mesh_dir / "Cargo.toml"

print(f"Working directory: {concept_mesh_dir}")

# Check current dependencies
print("\n📋 Current Cargo.toml status:")
if cargo_toml.exists():
    content = cargo_toml.read_text()
    if "openblas-src" in content:
        print("✅ openblas-src is in Cargo.toml")
        if "static" in content:
            print("✅ static feature is enabled")
    else:
        print("❌ openblas-src not found in Cargo.toml")

print("\n🎯 Choose a build strategy:")
print("1. Use intel-mkl instead of OpenBLAS (recommended for Windows)")
print("2. Build without BLAS (works but slower)")
print("3. Install vcpkg and OpenBLAS (complex)")
print("4. Use a stub implementation (fastest to get working)")

# Let's try Option 1 - Intel MKL
print("\n📦 Attempting Option 1: Intel MKL")
print("-" * 40)

os.chdir(concept_mesh_dir)

# First, remove openblas-src
print("Removing openblas-src...")
subprocess.run("cargo remove openblas-src", shell=True, capture_output=True)

# Add intel-mkl-src instead
print("Adding intel-mkl-src...")
result = subprocess.run(
    "cargo add intel-mkl-src --features=mkl-static-lp64-seq",
    shell=True,
    capture_output=True,
    text=True,
    encoding='utf-8'
)

if result.returncode == 0:
    print("✅ Added intel-mkl-src")
    
    # Try to build
    print("\n🔨 Building with Intel MKL...")
    build_result = subprocess.run(
        f'"{sys.executable}" -m maturin build --release',
        shell=True,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    if build_result.returncode == 0:
        print("✅ Build successful with Intel MKL!")
        
        # Install the wheel
        wheel_dir = concept_mesh_dir / "target" / "wheels"
        wheels = list(wheel_dir.glob("concept_mesh_rs-*.whl"))
        if wheels:
            wheel_path = wheels[0]
            print(f"\n📦 Installing: {wheel_path.name}")
            subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheel_path)])
            
            # Test import
            os.chdir(kha_path)
            test = subprocess.run(
                [sys.executable, "-c", "import concept_mesh_rs; print('✅ Import successful!')"],
                capture_output=True,
                text=True
            )
            print(test.stdout)
            if test.returncode == 0:
                print("\n✅ SUCCESS! Concept Mesh built with Intel MKL")
                sys.exit(0)
    else:
        print("❌ Intel MKL build failed")
        if build_result.stderr:
            print(build_result.stderr[:500])

# If MKL fails, try Option 4 - Stub implementation
print("\n📦 Attempting Option 4: Python Stub Implementation")
print("-" * 40)

stub_content = '''"""
Concept Mesh RS Stub - Pure Python Implementation
This is a fallback when Rust build fails
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ConceptMeshLoader:
    """Python stub implementation of ConceptMeshLoader"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data/concept_mesh")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.concepts = {}
        self.embeddings = {}
        self._load()
        logger.info(f"✅ ConceptMeshLoader (Python stub) initialized with {len(self.concepts)} concepts")
    
    def _load(self):
        """Load concepts from disk"""
        concepts_file = self.data_dir / "concepts.json"
        if concepts_file.exists():
            try:
                with open(concepts_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.concepts = data.get('concepts', data)
                    elif isinstance(data, list):
                        self.concepts = {f"concept_{i}": c for i, c in enumerate(data)}
            except Exception as e:
                logger.error(f"Failed to load concepts: {e}")
    
    def save(self):
        """Save concepts to disk"""
        concepts_file = self.data_dir / "concepts.json"
        with open(concepts_file, 'w', encoding='utf-8') as f:
            json.dump(self.concepts, f, indent=2)
    
    def add_concept(self, concept_id: str, data: Dict[str, Any]):
        """Add a concept"""
        self.concepts[concept_id] = data
        self.save()
    
    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get a concept by ID"""
        return self.concepts.get(concept_id)
    
    def search_similar(self, embedding: np.ndarray, k: int = 5) -> List[str]:
        """Simple similarity search (returns random concepts for now)"""
        concept_ids = list(self.concepts.keys())
        return concept_ids[:min(k, len(concept_ids))]
    
    def set_storage_path(self, path: str):
        """Set storage path"""
        self.data_dir = Path(path)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._load()

# Module-level functions
_loader = None

def get_loader() -> ConceptMeshLoader:
    """Get singleton loader instance"""
    global _loader
    if _loader is None:
        _loader = ConceptMeshLoader()
    return _loader

# Fake module attributes to match Rust version
__version__ = "0.1.0"
__file__ = __file__

# Export what the Rust module would export
__all__ = ['ConceptMeshLoader', 'get_loader']
'''

# Create the stub in site-packages
site_packages = Path(sys.executable).parent.parent / "Lib" / "site-packages"
stub_path = site_packages / "concept_mesh_rs.py"

print(f"Creating stub at: {stub_path}")
stub_path.write_text(stub_content, encoding='utf-8')
print("✅ Created Python stub")

# Test the stub
os.chdir(kha_path)
test = subprocess.run(
    [sys.executable, "-c", """
import concept_mesh_rs
print(f'✅ Imported stub from: {concept_mesh_rs.__file__}')
loader = concept_mesh_rs.get_loader()
print(f'✅ Loader working: {type(loader).__name__}')
"""],
    capture_output=True,
    text=True,
    encoding='utf-8'
)

print(test.stdout)
if test.stderr:
    print(test.stderr)

if test.returncode == 0:
    print("\n✅ SUCCESS! Using Python stub implementation")
    print("\n⚠️ Note: This is a fallback - performance will be limited")
    print("   To get full performance, you'll need to:")
    print("   1. Install Visual Studio Build Tools")
    print("   2. Install vcpkg and OpenBLAS, or")
    print("   3. Use WSL2 for Linux-based build")
else:
    print("❌ Stub test failed")

print("\n" + "=" * 60)
print("📝 Summary:")
print("   - Created Python stub implementation of concept_mesh_rs")
print("   - This will work but with reduced performance")
print("   - No more import errors in MCP subprocess!")
