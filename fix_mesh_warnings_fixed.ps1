# Fix concept_mesh installation and warnings

Write-Host "🔧 Starting concept_mesh fix..." -ForegroundColor Green

# 1. Turn concept_mesh into a package and install it editable
Set-Location "C:\Users\jason\Desktop\tori\kha\concept_mesh"

# Create __init__.py if it doesn't exist
if (!(Test-Path "__init__.py")) {
    Write-Host "Creating __init__.py in concept_mesh..." -ForegroundColor Yellow
    "" | Out-File -Encoding utf8 "__init__.py"
}

# Install concept_mesh as editable package
Write-Host "Installing concept_mesh as editable package..." -ForegroundColor Yellow
& C:\ALANPY311\Scripts\pip.exe install -e .

# 2. Sanity-check from the MCP folder
Set-Location "..\mcp_metacognitive"

Write-Host "`n🔍 Testing concept_mesh import..." -ForegroundColor Green

# Create a temporary Python script for testing
$testScript = @"
import importlib.util
import os
import sys

print("CWD:", os.getcwd())
print("Python:", sys.executable)
print("sys.path includes:")
for p in sys.path[:5]:
    print(f"  - {p}")

# Test concept_mesh import
spec = importlib.util.find_spec("concept_mesh")
print("\nconcept_mesh found:", bool(spec))
if spec:
    print("Location:", spec.origin)
else:
    print("Location: NOT FOUND")

# Try to actually import it
try:
    import concept_mesh
    print("\n✅ concept_mesh imported successfully!")
    
    # Check for interface module
    try:
        from concept_mesh.interface import ConceptMesh
        print("✅ ConceptMesh interface available!")
    except ImportError as e:
        print(f"❌ ConceptMesh interface not available: {e}")
        
except ImportError as e:
    print(f"\n❌ Failed to import concept_mesh: {e}")

# Test soliton_memory import
print("\n🔍 Testing soliton_memory import...")
try:
    from mcp_metacognitive.core import soliton_memory
    print("✅ soliton_memory imported successfully!")
    
    # Test specific functions
    if hasattr(soliton_memory, 'initialize_user'):
        print("✅ initialize_user function available!")
    if hasattr(soliton_memory, 'get_user_stats'):
        print("✅ get_user_stats function available!")
    if hasattr(soliton_memory, 'check_health'):
        print("✅ check_health function available!")
        
except ImportError as e:
    print(f"❌ Failed to import soliton_memory: {e}")
"@

# Save the test script
$testScript | Out-File -Encoding utf8 "test_imports_temp.py"

# Run the test script
& C:\ALANPY311\python.exe test_imports_temp.py

# Clean up
Remove-Item "test_imports_temp.py" -ErrorAction SilentlyContinue

Write-Host "`n✅ Fix script completed!" -ForegroundColor Green
