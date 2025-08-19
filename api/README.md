# API Directory Structure

## ⚠️ IMPORTANT: File Status ⚠️

### ✅ ACTIVE FILES (Currently in use):
- `routes/soliton.py` - Active soliton memory routes (imported by prajna_api.py)
- `routes/concept_mesh.py` - Concept mesh routes
- `routes/hologram.py` - Hologram routes
- Other files in `routes/` directory

### ❌ DEPRECATED FILES (DO NOT USE):
- `soliton.py.deprecated` - Old soliton implementation (replaced by routes/soliton.py)
- `main.py` - Old main API file (replaced by prajna/api/prajna_api.py)
- `soliton_route.py` - Old alias file (no longer needed)

### 📝 Notes:
The actual API is served through:
- **Main API**: `prajna/api/prajna_api.py` (launched by enhanced_launcher.py)
- **Routes**: Files in the `routes/` subdirectory
- **NOT**: Files directly in this directory (most are deprecated)

### 🚀 Current Flow:
1. `enhanced_launcher.py` starts the server
2. It runs `prajna.api.prajna_api:app` 
3. `prajna_api.py` imports routes from `api.routes.soliton`
4. This loads `api/routes/soliton.py` (the active file)

### 🧹 Cleanup TODO:
- Remove deprecated files after confirming system stability
- Move any useful code from deprecated files to active ones
- Update any remaining imports that reference deprecated files
