# Import Order Fix for Concept Mesh

To ensure imports always try Rust → stub (addressing the yellow item in audit):

## In `mcp_metacognitive/core/soliton_memory.py`

Update the import section to explicitly try in order:

```python
# Import concept mesh - try Rust wheel first, then stub
CONCEPT_MESH_AVAILABLE = False
ConceptMesh = None

# 1. Try the compiled Rust wheel (concept-mesh-rs)
try:
    from concept_mesh import ConceptMesh as RustConceptMesh
    from concept_mesh.types import MemoryEntry, MemoryQuery, PhaseTag
    ConceptMesh = RustConceptMesh
    CONCEPT_MESH_AVAILABLE = True
    logger.info("✅ Using Rust concept_mesh implementation")
except ImportError:
    # 2. Try the Python stub (concept_mesh)
    try:
        from concept_mesh_stub import ConceptMesh as StubConceptMesh
        from concept_mesh_stub.types import MemoryEntry, MemoryQuery, PhaseTag
        ConceptMesh = StubConceptMesh
        CONCEPT_MESH_AVAILABLE = True
        logger.warning("⚠️ Using Python stub for concept_mesh")
    except ImportError:
        # 3. Fall back to inline mock
        logger.warning("❌ No concept_mesh implementation available - using mock")
        CONCEPT_MESH_AVAILABLE = False
```

## In any other files importing concept_mesh:

```python
# Standard import order
try:
    # Try Rust implementation first
    from concept_mesh import SomeClass
except ImportError:
    try:
        # Fall back to stub
        from concept_mesh_stub import SomeClass
    except ImportError:
        # Use mock or raise
        raise ImportError("No concept_mesh implementation available")
```

This ensures:
1. Rust wheel is preferred when available (fastest)
2. Python stub is used as fallback (for development)
3. Mock is last resort (for testing without either)
