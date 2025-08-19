# Concept Mesh Import Order Documentation

## Namespace Resolution (v0.12.0+)

To prevent namespace collisions, we use the following import pattern:

### 1. Rust Wheel (Production)
- **Package Name**: `concept_mesh_rs` 
- **Import Name**: `concept_mesh_rs`
- **Built From**: `concept-mesh/` Rust crate
- **Install**: `pip install concept_mesh_rs` (from wheel)

### 2. Python Stub (Development)
- **Package Name**: `concept_mesh`
- **Import Name**: `concept_mesh`
- **Location**: `concept_mesh/` Python directory
- **Purpose**: Development fallback when Rust wheel unavailable

## Standard Import Pattern

```python
# In all modules that need concept mesh:
try:
    # Try production Rust wheel first
    import concept_mesh_rs as cm
    CONCEPT_MESH_AVAILABLE = True
    USING_STUB = False
except ImportError:
    try:
        # Fall back to Python stub for development
        from concept_mesh import ConceptMeshStub as cm
        CONCEPT_MESH_AVAILABLE = True
        USING_STUB = True
    except ImportError:
        # No concept mesh available
        CONCEPT_MESH_AVAILABLE = False
        USING_STUB = False
        cm = None
```

## File Locations

- **Rust Source**: `concept-mesh/src/lib.rs`
- **Cargo.toml**: Update `name = "concept_mesh_rs"`
- **Python Stub**: `concept_mesh/__init__.py`
- **CI Build**: `.github/workflows/build-concept-mesh.yml`

## Migration Checklist

1. ✅ Update `concept-mesh/Cargo.toml`:
   ```toml
   [package]
   name = "concept_mesh_rs"
   ```

2. ✅ Update all imports in:
   - `api/routes/soliton.py`
   - `mcp_metacognitive/core/soliton_memory.py`
   - Any other files using concept mesh

3. ✅ Update CI workflow to build `concept_mesh_rs` wheel

4. ✅ Update installation docs to use `pip install concept_mesh_rs`

## Why This Pattern?

- **No Shadowing**: Clear distinction between Rust (`_rs`) and Python packages
- **Graceful Fallback**: Development works without Rust toolchain
- **Type Safety**: Both implementations share same interface
- **CI Friendly**: Tests can run with or without compiled wheel
