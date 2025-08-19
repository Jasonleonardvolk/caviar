"""
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
