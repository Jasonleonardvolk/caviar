#!/usr/bin/env python3
"""
Fix concept mesh import issues
Ensures all modules can find the concept mesh implementations
"""

import sys
import os
from pathlib import Path

print("🔧 FIXING CONCEPT MESH IMPORT ISSUES")
print("=" * 60)

# Get the kha directory path
kha_path = Path(__file__).parent.absolute()
print(f"Working directory: {kha_path}")

# Fix 1: Ensure concept_mesh_rs.py is in the right place
print("\n📌 Fix 1: Checking concept_mesh_rs.py...")
print("-" * 40)

concept_mesh_rs_file = kha_path / "concept_mesh_rs.py"
if concept_mesh_rs_file.exists():
    print("✅ concept_mesh_rs.py exists in root directory")
else:
    print("❌ concept_mesh_rs.py not found!")

# Fix 2: Create a proper Python stub if needed
print("\n📌 Fix 2: Creating concept mesh stub...")
print("-" * 40)

# Check if we need to create the stub in python/core
python_core_stub = kha_path / "python" / "core" / "concept_mesh_stub.py"
python_core_stub.parent.mkdir(parents=True, exist_ok=True)

stub_content = '''"""
Concept Mesh Stub - Fallback implementation
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading

logger = logging.getLogger(__name__)

class ConceptMeshStub:
    """Minimal concept mesh implementation for fallback"""
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, config: Dict[str, Any] = None):
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config or {})
        return cls._instance
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_path = Path(config.get('storage_path', 'data/concept_mesh'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.concepts = {}
        self.concept_file = self.storage_path / 'concepts.json'
        
        # Load existing concepts
        if self.concept_file.exists():
            try:
                with open(self.concept_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.concepts = data.get('concepts', {})
                    elif isinstance(data, list):
                        # Convert list to dict
                        self.concepts = {f"concept_{i}": c for i, c in enumerate(data)}
            except Exception as e:
                logger.error(f"Failed to load concepts: {e}")
        
        logger.info(f"✅ ConceptMeshStub initialized with {len(self.concepts)} concepts")
    
    def add_concept(self, concept_id: str, name: str, **kwargs):
        """Add a concept to the mesh"""
        self.concepts[concept_id] = {
            'id': concept_id,
            'name': name,
            **kwargs
        }
        self._save()
        return concept_id
    
    def get_concept(self, concept_id: str):
        """Get a concept by ID"""
        return self.concepts.get(concept_id)
    
    def search_concepts(self, query: str, limit: int = 10):
        """Simple search by name"""
        results = []
        query_lower = query.lower()
        
        for concept_id, concept in self.concepts.items():
            if query_lower in concept.get('name', '').lower():
                results.append(concept)
                if len(results) >= limit:
                    break
        
        return results
    
    def _save(self):
        """Save concepts to disk"""
        try:
            data = {
                'concepts': self.concepts,
                'version': '1.0'
            }
            with open(self.concept_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save concepts: {e}")

# Global instance getter
def get_mesh_instance(config: Dict[str, Any] = None):
    """Get the concept mesh instance"""
    return ConceptMeshStub.get_instance(config)
'''

with open(python_core_stub, 'w', encoding='utf-8') as f:
    f.write(stub_content)
print(f"✅ Created concept mesh stub: {python_core_stub}")

# Fix 3: Update PYTHONPATH to include necessary directories
print("\n📌 Fix 3: Setting up Python paths...")
print("-" * 40)

# Create a sitecustomize.py if it doesn't exist
sitecustomize_file = kha_path / "sitecustomize.py"
if not sitecustomize_file.exists():
    sitecustomize_content = f'''"""
Site customization to ensure proper imports
"""
import sys
from pathlib import Path

# Add kha directory to path
kha_path = Path(__file__).parent
if str(kha_path) not in sys.path:
    sys.path.insert(0, str(kha_path))

# Add python/core to path
python_core = kha_path / "python" / "core"
if str(python_core) not in sys.path:
    sys.path.insert(0, str(python_core))

print(f"✅ Python paths configured: {{sys.path[:2]}}")
'''
    
    with open(sitecustomize_file, 'w', encoding='utf-8') as f:
        f.write(sitecustomize_content)
    print("✅ Created sitecustomize.py for import configuration")
else:
    print("✅ sitecustomize.py already exists")

# Fix 4: Create a wrapper that handles all import fallbacks
print("\n📌 Fix 4: Creating unified concept mesh wrapper...")
print("-" * 40)

wrapper_file = kha_path / "concept_mesh_wrapper.py"
wrapper_content = '''"""
Unified Concept Mesh Wrapper
Handles all import fallbacks gracefully
"""
import logging

logger = logging.getLogger(__name__)

# Try imports in order of preference
mesh_instance = None
backend_type = "none"

# 1. Try Rust implementation
try:
    import concept_mesh_rs
    mesh_instance = concept_mesh_rs.get_loader()
    backend_type = "rust"
    logger.info("✅ Using Rust concept mesh implementation")
except ImportError:
    pass

# 2. Try Python core implementation
if mesh_instance is None:
    try:
        from python.core.concept_mesh import ConceptMesh
        mesh_instance = ConceptMesh()
        backend_type = "python_core"
        logger.info("✅ Using Python core concept mesh")
    except ImportError:
        pass

# 3. Try stub implementation
if mesh_instance is None:
    try:
        from python.core.concept_mesh_stub import get_mesh_instance
        mesh_instance = get_mesh_instance()
        backend_type = "stub"
        logger.info("✅ Using concept mesh stub")
    except ImportError:
        pass

# 4. Final fallback - minimal mock
if mesh_instance is None:
    logger.warning("⚠️ No concept mesh implementation found - using mock")
    
    class MockMesh:
        def add_concept(self, *args, **kwargs):
            return "mock_id"
        
        def get_concept(self, *args, **kwargs):
            return {}
        
        def search_concepts(self, *args, **kwargs):
            return []
    
    mesh_instance = MockMesh()
    backend_type = "mock"

# Export
def get_concept_mesh():
    """Get the active concept mesh instance"""
    return mesh_instance

def get_backend_type():
    """Get the type of backend being used"""
    return backend_type

__all__ = ['get_concept_mesh', 'get_backend_type']
'''

with open(wrapper_file, 'w', encoding='utf-8') as f:
    f.write(wrapper_content)
print(f"✅ Created concept mesh wrapper: {wrapper_file}")

print("\n" + "=" * 60)
print("✅ Import fixes applied!")
print("\n📝 The system will now:")
print("   1. Try to import Rust concept_mesh_rs")
print("   2. Fall back to Python core implementation")
print("   3. Fall back to stub implementation")
print("   4. Use mock as last resort")
print("\n🔧 To use in your code:")
print("   from concept_mesh_wrapper import get_concept_mesh")
print("   mesh = get_concept_mesh()")
print("\n🚀 Restart the server to apply changes!")
