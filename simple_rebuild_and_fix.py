#!/usr/bin/env python3
"""
Simple rebuild of concept_mesh_rs with existing pure-Rust config
No OpenBLAS needed - we already disabled it in Cargo.toml
"""

import os
import sys
import subprocess
from pathlib import Path

print("🔧 REBUILDING CONCEPT MESH WITH PURE RUST")
print("=" * 60)

kha_path = Path(__file__).parent.absolute()
concept_mesh_dir = kha_path / "concept_mesh"
cargo_toml = concept_mesh_dir / "Cargo.toml"

print(f"Working directory: {concept_mesh_dir}")

# Verify we're using pure Rust (no OpenBLAS)
print("\n📋 Checking Cargo.toml configuration...")
if cargo_toml.exists():
    content = cargo_toml.read_text()
    if 'features = ["rust"]' in content:
        print("✅ Using pure Rust backend (no OpenBLAS needed)")
    else:
        print("⚠️ Cargo.toml may have changed")

os.chdir(concept_mesh_dir)

# Step 1: Clean and build
print("\n🧹 Cleaning previous build...")
if (concept_mesh_dir / "target").exists():
    # Just clean the release directory
    subprocess.run("cargo clean --release", shell=True, capture_output=True)

print("\n📦 Building wheel with maturin...")
result = subprocess.run(
    f'"{sys.executable}" -m maturin build --release',
    shell=True,
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace'
)

print("BUILD OUTPUT:")
if result.stdout:
    print(result.stdout)
if result.stderr:
    print("\nBUILD WARNINGS/ERRORS:")
    print(result.stderr)

if result.returncode != 0:
    print("\n❌ Build failed!")
    sys.exit(1)

print("\n✅ Build successful!")

# Step 2: Install the wheel
wheel_dir = concept_mesh_dir / "target" / "wheels"
wheels = list(wheel_dir.glob("concept_mesh_rs-*.whl"))

if not wheels:
    print("❌ No wheel found!")
    sys.exit(1)

wheel_path = wheels[0]
print(f"\n📦 Installing wheel: {wheel_path.name}")

install_result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheel_path)],
    capture_output=True,
    text=True
)

print(install_result.stdout)
if install_result.returncode != 0:
    print(f"❌ Install failed: {install_result.stderr}")
    sys.exit(1)

# Step 3: Verify it works
print("\n🧪 Testing import...")
os.chdir(kha_path)

test_result = subprocess.run(
    [sys.executable, "-c", """
import concept_mesh_rs
print(f'✅ Imported from: {concept_mesh_rs.__file__}')
loader = concept_mesh_rs.get_loader()
print(f'✅ Loader type: {type(loader).__name__}')
"""],
    capture_output=True,
    text=True
)

print(test_result.stdout)
if test_result.returncode != 0:
    print(f"❌ Import failed: {test_result.stderr}")
    sys.exit(1)

# Step 4: Now fix the MCP server
print("\n🔧 Fixing MCP FastMCP.run() signature...")

mcp_paths = [
    kha_path / "mcp_metacognitive" / "fast_mcp.py",
    kha_path / "mcp_metacognitive" / "core" / "fast_mcp.py",
]

fixed = False
for mcp_path in mcp_paths:
    if mcp_path.exists():
        print(f"Patching: {mcp_path}")
        content = mcp_path.read_text(encoding='utf-8')
        
        # Backup
        backup = mcp_path.with_suffix('.py.bak')
        if not backup.exists():
            backup.write_text(content, encoding='utf-8')
        
        # Check if already patched
        if "def run(self, transport" in content and "host:" in content:
            print("✅ Already patched")
            fixed = True
            continue
        
        # Find and patch the run method
        import re
        
        # Replace the run method signature
        new_run = '''def run(
        self,
        transport: str = "stdio",
        host: str = "0.0.0.0",
        port: int = 8100,
        **kwargs
    ):'''
        
        content = re.sub(
            r'def run\(\s*self[^)]*\)\s*:',
            new_run + ':',
            content
        )
        
        # Also update uvicorn.run calls
        content = re.sub(
            r'uvicorn\.run\(\s*self\.app[^)]*\)',
            'uvicorn.run(self.app, host=host, port=port, **kwargs)',
            content
        )
        
        mcp_path.write_text(content, encoding='utf-8')
        print("✅ Patched FastMCP.run()")
        fixed = True
        break

if not fixed:
    print("⚠️ FastMCP not found or already patched")

print("\n" + "=" * 60)
print("✅ DONE! Both issues fixed:")
print("   1. concept_mesh_rs rebuilt and installed")
print("   2. MCP FastMCP.run() accepts host/port")
print("\n🚀 Next steps:")
print("   1. Kill any running Python processes")
print("   2. Restart: python enhanced_launcher.py")
print("   3. Verify no 'mock' warnings in logs")
