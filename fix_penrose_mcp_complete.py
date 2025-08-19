#!/usr/bin/env python3
"""
Fix Penrose Concept Mesh and MCP Server Issues
Two-track repair: Concept Mesh imports + MCP run() signature
"""

import sys
import subprocess
import os
from pathlib import Path
import shutil

print("🔧 FIXING PENROSE CONCEPT MESH & MCP SERVER")
print("=" * 60)

# Get paths
kha_path = Path(__file__).parent.absolute()
concept_mesh_dir = kha_path / "concept_mesh"
venv_python = sys.executable

print(f"Working directory: {kha_path}")
print(f"Python executable: {venv_python}")

# TRACK 1: Fix Penrose Concept Mesh
print("\n🦀 TRACK 1: Building Penrose Concept Mesh Wheel")
print("-" * 40)

# Step 1.1: Build the wheel
print("\n📦 Step 1.1: Building concept_mesh_rs wheel...")
if concept_mesh_dir.exists():
    os.chdir(concept_mesh_dir)
    
    # Install maturin
    print("Installing maturin...")
    subprocess.run([venv_python, "-m", "pip", "install", "-U", "maturin"], check=True)
    
    # Build the wheel
    print("Building release wheel...")
    result = subprocess.run([venv_python, "-m", "maturin", "build", "--release"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"❌ Build failed: {result.stderr}")
        print("\n💡 If OpenBLAS error on Windows, run:")
        print("   cargo add openblas-src --features=static")
        print("   Then retry this script")
    else:
        # Install the wheel
        wheel_dir = concept_mesh_dir / "target" / "wheels"
        wheels = list(wheel_dir.glob("concept_mesh_rs-*.whl"))
        if wheels:
            wheel_path = wheels[0]
            print(f"Installing wheel: {wheel_path}")
            subprocess.run([venv_python, "-m", "pip", "install", wheel_path], check=True)
            print("✅ Wheel installed successfully!")
        else:
            print("❌ No wheel found after build")
    
    os.chdir(kha_path)
else:
    print("❌ concept_mesh directory not found!")

# Step 1.2: Create/update concept_mesh_wrapper.py
print("\n📝 Step 1.2: Creating fail-fast wrapper...")
wrapper_path = kha_path / "core" / "concept_mesh_wrapper.py"
wrapper_path.parent.mkdir(exist_ok=True)

wrapper_content = '''"""
Concept Mesh Wrapper - Fail-fast if Penrose not available
"""
import importlib
import sys

try:
    cm = importlib.import_module("concept_mesh_rs")
except ImportError as e:
    raise RuntimeError(
        "❌ Penrose native module missing – run maturin build in concept_mesh/"
    ) from e

# Expose a singleton
_mesh = cm.ConceptMeshLoader()

def get_mesh(path: str = None):
    """Get the concept mesh instance, optionally with custom storage path"""
    if path:
        cm.set_storage_path(path)
    return _mesh

def get_backend():
    """Return the backend type"""
    return "penrose_rust"

__all__ = ['get_mesh', 'get_backend']
'''

wrapper_path.write_text(wrapper_content, encoding='utf-8')
print(f"✅ Created wrapper: {wrapper_path}")

# Step 1.3: Fix subprocess spawning in MCP
print("\n🔧 Step 1.3: Fixing MCP subprocess spawning...")
mcp_server_path = kha_path / "mcp_metacognitive" / "server.py"
if mcp_server_path.exists():
    content = mcp_server_path.read_text(encoding='utf-8')
    
    # Backup
    backup_path = mcp_server_path.with_suffix('.py.bak_penrose_fix')
    if not backup_path.exists():
        backup_path.write_text(content, encoding='utf-8')
        print(f"✅ Backed up: {backup_path}")
    
    # Replace subprocess.Popen(['python', ...]) with sys.executable
    if "subprocess.Popen(['python'" in content or 'subprocess.Popen(["python"' in content:
        import re
        content = re.sub(
            r'subprocess\.Popen\(\[[\'"](python)[\'"]',
            'subprocess.Popen([sys.executable',
            content
        )
        mcp_server_path.write_text(content, encoding='utf-8')
        print("✅ Fixed subprocess to use sys.executable")
    else:
        print("✅ Subprocess already uses correct Python")

# TRACK 2: Fix MCP run() signature
print("\n🔧 TRACK 2: Fixing MCP run() Signature")
print("-" * 40)

# Find FastMCP file
fast_mcp_paths = [
    kha_path / "mcp_metacognitive" / "fast_mcp.py",
    kha_path / "mcp_metacognitive" / "core" / "fast_mcp.py",
]

for fast_mcp_path in fast_mcp_paths:
    if fast_mcp_path.exists():
        print(f"\n📝 Patching FastMCP in: {fast_mcp_path}")
        content = fast_mcp_path.read_text(encoding='utf-8')
        
        # Backup
        backup_path = fast_mcp_path.with_suffix('.py.bak_run_fix')
        if not backup_path.exists():
            backup_path.write_text(content, encoding='utf-8')
            print(f"✅ Backed up: {backup_path}")
        
        # Check if run() needs patching
        if "def run(self" in content and "host:" not in content:
            # Find the run method and patch it
            import re
            
            # Pattern to find the run method
            pattern = r'(def run\(self[^:]*\):[^\n]*)'
            
            replacement = '''def run(
        self,
        transport: str = "stdio",
        host: str = "0.0.0.0",
        port: int = 8100,
        **uvicorn_kwargs,
    ):'''
            
            new_content = re.sub(pattern, replacement, content)
            
            # Also need to update the method body to use these params
            if "uvicorn.run(" in new_content:
                # Find uvicorn.run calls and update them
                new_content = re.sub(
                    r'uvicorn\.run\(\s*self\.app[^)]*\)',
                    '''uvicorn.run(
                self.app,
                host=host,
                port=port,
                log_level="info",
                **uvicorn_kwargs,
            )''',
                    new_content
                )
            
            fast_mcp_path.write_text(new_content, encoding='utf-8')
            print("✅ Patched FastMCP.run() to accept host/port")
        else:
            print("✅ FastMCP.run() already accepts host/port or not found")
        break
else:
    print("❌ FastMCP file not found in expected locations")

# Verification
print("\n✅ VERIFICATION")
print("-" * 40)

# Test import
print("\n1. Testing concept_mesh_rs import:")
test_result = subprocess.run(
    [venv_python, "-c", "import concept_mesh_rs; print(f'✅ Imported from: {concept_mesh_rs.__file__}')"],
    capture_output=True,
    text=True
)
print(test_result.stdout)
if test_result.returncode != 0:
    print(f"❌ Import failed: {test_result.stderr}")

print("\n" + "=" * 60)
print("✅ Fixes applied!")
print("\n🚀 Next steps:")
print("   1. Stop any running servers (taskkill /IM python.exe /F)")
print("   2. Restart: python enhanced_launcher.py")
print("   3. Check logs for:")
print("      - '🦀 Penrose backend: rust' (main log)")
print("      - NO 'using mock' warnings in MCP stderr")
print("      - 'Uvicorn running on http://0.0.0.0:8100'")
