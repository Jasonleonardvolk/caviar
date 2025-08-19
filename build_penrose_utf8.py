#!/usr/bin/env python3
"""
Build Penrose with proper encoding handling
"""

import os
import sys
import subprocess
from pathlib import Path

print("🔧 BUILDING PENROSE WITH UTF-8 ENCODING")
print("=" * 60)

# Get paths
kha_path = Path(__file__).parent.absolute()
concept_mesh_dir = kha_path / "concept_mesh"

# Set UTF-8 encoding for subprocess
env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'
env['LANG'] = 'en_US.UTF-8'

print(f"Working directory: {concept_mesh_dir}")
os.chdir(concept_mesh_dir)

# Step 1: Build with maturin
print("\n📦 Building concept_mesh_rs wheel with UTF-8...")

# Use shell=True on Windows to properly handle encoding
if sys.platform == "win32":
    build_cmd = f'"{sys.executable}" -m maturin build --release'
    result = subprocess.run(
        build_cmd,
        shell=True,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
else:
    result = subprocess.run(
        [sys.executable, "-m", "maturin", "build", "--release"],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )

# Print output regardless of success/failure
if result.stdout:
    print("STDOUT:")
    print(result.stdout)
if result.stderr:
    print("STDERR:")
    print(result.stderr)

if result.returncode != 0:
    print("\n❌ Build failed!")
    
    # Check for common issues
    if result.stderr and ("openblas" in result.stderr.lower() or "lapack" in result.stderr.lower()):
        print("\n💡 OpenBLAS/LAPACK issue detected!")
        print("   Run these commands:")
        print("   cd concept_mesh")
        print("   cargo add openblas-src --features=static")
        print("   Then run this script again")
    elif result.stderr and "error: Microsoft Visual C++" in result.stderr:
        print("\n💡 Visual C++ issue detected!")
        print("   Install Visual Studio Build Tools:")
        print("   https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022")
    elif result.stderr and "error: linker `link.exe` not found" in result.stderr:
        print("\n💡 Linker not found!")
        print("   Install Visual Studio Build Tools or run from VS Developer Command Prompt")
    else:
        print("\n💡 Check the error output above for details")
    
    sys.exit(1)

print("✅ Build successful!")

# Step 2: Install the wheel
wheel_dir = concept_mesh_dir / "target" / "wheels"
wheels = list(wheel_dir.glob("concept_mesh_rs-*.whl"))

if not wheels:
    print("❌ No wheel found after build")
    sys.exit(1)

wheel_path = wheels[0]
print(f"\n📦 Installing wheel: {wheel_path.name}")

install_result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheel_path)],
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace'
)

print(install_result.stdout)
if install_result.returncode != 0:
    print(f"❌ Install failed: {install_result.stderr}")
    sys.exit(1)

print("✅ Wheel installed successfully!")

# Step 3: Test import
print("\n🧪 Testing import...")
os.chdir(kha_path)

test_result = subprocess.run(
    [sys.executable, "-c", "import concept_mesh_rs; print(f'✅ concept_mesh_rs imported from: {concept_mesh_rs.__file__}')"],
    capture_output=True,
    text=True,
    encoding='utf-8'
)

print(test_result.stdout)
if test_result.returncode != 0:
    print(f"❌ Import test failed: {test_result.stderr}")
    print("\n💡 Try running:")
    print(f"   {sys.executable} -c \"import concept_mesh_rs\"")
    print("   to see the full error")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ Penrose wheel built and installed successfully!")
print("\n🚀 Next steps:")
print("   1. Run: python fix_penrose_mcp_complete.py")
print("   2. Restart the server")
print("   3. Check that concept mesh loads without 'mock' warnings")
