# CRITICAL CONCEPT MESH INFORMATION
**Last Updated**: January 2025

## 🎯 How to Build concept_mesh_rs Wheel

### Location
- **Rust source**: `${IRIS_ROOT}\concept_mesh\`
- **Build output**: `${IRIS_ROOT}\concept_mesh\target\wheels\`
- **Installed to**: `.venv\Lib\site-packages\concept_mesh_rs.cp311-win_amd64.pyd`

### Build Commands
```bash
# From ${IRIS_ROOT}
cd concept_mesh
maturin build --release
pip install target\wheels\concept_mesh_rs-*.whl
```

### Quick Build Script
```bash
python build_wheel_onepager.py
```

## 📁 Where Concept Mesh Data is Stored

### Primary Data Locations
1. **User/Group Scoped Data**: 
   - `${IRIS_ROOT}\data\memory_vault\`
   - `${IRIS_ROOT}\data\groups\{group_id}\concepts.json`
   - `${IRIS_ROOT}\data\users\{user_id}\concepts.json`

2. **Global Concept Mesh Data**:
   - `${IRIS_ROOT}\data\concept_mesh\concepts.json`
   - `${IRIS_ROOT}\concept_mesh_diffs.json`
   - `${IRIS_ROOT}\concept_mesh_data.json`

3. **Soliton Memory Data**:
   - `${IRIS_ROOT}\data\soliton_mesh\`
   - `${IRIS_ROOT}\fractal_soliton_memory.pkl.gz`

### Data Structure Format
```json
{
  "concepts": {
    "concept_id": {
      "id": "concept_id",
      "name": "Concept Name",
      "embedding": [...],
      "metadata": {}
    }
  },
  "version": "1.0",
  "metadata": {
    "created": "2024-01-01T00:00:00Z",
    "last_updated": "2024-01-01T00:00:00Z",
    "count": 0
  }
}
```

## 🔧 Common Issues & Fixes

### "Neither concept_mesh_rs nor stub available"
1. Wheel not built/installed
2. Solution: Run `python build_wheel_onepager.py`

### "0 entries" errors
1. Data files missing or invalid
2. Solution: Run `python init_concept_mesh_data_simple.py`

### Import works in main but not MCP
1. MCP using different Python
2. Solution: Ensure MCP uses `sys.executable` not hardcoded "python"

## 🚀 Complete Setup Sequence
```bash
# 1. Activate venv
cd ${IRIS_ROOT}
.\.venv\Scripts\activate

# 2. Build and install wheel
python build_wheel_onepager.py

# 3. Initialize data files
python init_concept_mesh_data_simple.py

# 4. Start server
taskkill /IM python.exe /F
python enhanced_launcher.py
```

## 📝 Key Files
- **Cargo.toml**: Must have `ndarray-linalg = { default-features = false, features = ["rust"] }`
- **No OpenBLAS**: Using pure Rust implementation
- **Remove shadows**: Delete any `concept_mesh_rs.py` file in root

## Memory Architecture
- **UnifiedMemoryVault**: `python/core/memory_vault.py`
- **FractalSolitonMemory**: `python/core/fractal_soliton_memory.py`
- **ConceptMesh**: `python/core/concept_mesh.py`
- **Groups**: Organizations in `ingest_pdf/multi_tenant_manager.py`
