# Cognitive Interface & Concept Mesh Setup Guide

This guide helps you set up the `cognitive_interface` and `concept_mesh` services for the KHA project.

## Quick Start

1. **Run the quick setup script:**
   ```batch
   quick_setup_imports.bat
   ```

2. **Start the Cognitive Interface service:**
   ```batch
   start_cognitive_interface.bat
   ```

3. **Verify the service is running:**
   - Open http://localhost:5173/docs in your browser
   - You should see the FastAPI documentation page

## Manual Setup Steps

### 1. Setting up cognitive_interface

The `cognitive_interface` module is part of your project under `ingest_pdf/cognitive_interface.py`.

**Make it importable:**

```batch
# From project root (${IRIS_ROOT})
set PYTHONPATH=%cd%
```

**Ensure package structure:**
- Make sure `ingest_pdf/__init__.py` exists (even if empty)
- The quick_setup_imports.bat script creates this automatically

**Start the service:**
```batch
python -m uvicorn ingest_pdf.cognitive_interface:app --port 5173
```

### 2. Setting up concept_mesh

The `concept_mesh` service needs to be either:
- Installed from PyPI (if available)
- Cloned from your internal repository
- Mocked for testing

**Option A: Install from PyPI**
```batch
pip install concept-mesh-client
```

**Option B: Clone from internal repo**
```batch
git clone git@github.com:your-org/concept-mesh.git
cd concept-mesh
pip install -e .
```

**Option C: Use the mock (created by setup script)**
The setup script creates a basic mock at `concept_mesh/__init__.py`

**Start the mesh service (if you have the real one):**
```batch
uvicorn concept_mesh.api:app --port 8003
```

**Set environment variable:**
```batch
set CONCEPT_MESH_URL=http://localhost:8003/api/mesh
```

## Testing Your Setup

### Python REPL Test
```python
# Test 1: Import cognitive_interface
from ingest_pdf.cognitive_interface import add_concept_diff
print("✓ cognitive_interface imported successfully")

# Test 2: Import concept_mesh
from concept_mesh import ConceptMeshConnector
print("✓ concept_mesh imported successfully")

# Test 3: Create connector instance
connector = ConceptMeshConnector()
print("✓ ConceptMeshConnector created")
```

### Run your tests
```batch
pytest --maxfail=1 --disable-warnings -q
```

## Troubleshooting

### Import Error: No module named 'cognitive_interface'
- Make sure PYTHONPATH is set: `set PYTHONPATH=${IRIS_ROOT}`
- Check that `ingest_pdf/__init__.py` exists
- Try the alternative import: `from ingest_pdf.cognitive_interface import ...`

### Import Error: No module named 'concept_mesh'
- Install it: `pip install concept-mesh-client`
- Or use the mock created by the setup script
- Check if you need to clone an internal repository

### Port 5173 already in use
- The start script automatically kills existing processes
- Or manually: `taskkill /F /IM python.exe` (be careful, this kills ALL Python processes)

### Service not responding
- Check the console output for errors
- Verify the URL: http://localhost:5173/docs
- Check firewall settings

## File Structure

Your project should have this structure:
```
${IRIS_ROOT}\
├── ingest_pdf/
│   ├── __init__.py              # Required for Python package
│   ├── cognitive_interface.py   # Your FastAPI app
│   └── ...
├── concept_mesh/                # Mock or real module
│   └── __init__.py
├── quick_setup_imports.bat      # Quick setup script
├── start_cognitive_interface.bat # Start service script
├── setup_cognitive_and_mesh.py  # Comprehensive setup script
└── README_SETUP.md             # This file
```

## Advanced Setup

For a more comprehensive setup with health checks and automatic service management, run:
```batch
python setup_cognitive_and_mesh.py
```

This script:
- Automatically sets up Python paths
- Verifies package structure
- Starts services with health checks
- Creates convenient batch files
- Runs comprehensive tests

## Next Steps

Once both services are running:
1. Your imports should work without errors
2. The cognitive interface API will be available at http://localhost:5173
3. The concept mesh API (if running) will be at http://localhost:8003
4. You can proceed with your pipeline development

## Need Help?

If you're still having issues:
1. Check the console output for specific error messages
2. Verify all file paths are correct
3. Ensure Python and pip are properly installed
4. Check that uvicorn is installed: `pip install uvicorn[standard]`
