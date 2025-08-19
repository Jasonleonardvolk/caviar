#!/usr/bin/env python3
"""
Fix OpenBLAS dependency and build Penrose
"""

import os
import sys
import subprocess
from pathlib import Path

print("🔧 FIXING OPENBLAS AND BUILDING PENROSE")
print("=" * 60)

# Get paths
kha_path = Path(__file__).parent.absolute()
concept_mesh_dir = kha_path / "concept_mesh"

# Set UTF-8 encoding
env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'

print(f"Working directory: {concept_mesh_dir}")
os.chdir(concept_mesh_dir)

# Step 1: Add openblas-src with static feature
print("\n📦 Step 1: Adding openblas-src with static feature...")

cargo_cmd = "cargo add openblas-src --features=static"
print(f"Running: {cargo_cmd}")

result = subprocess.run(
    cargo_cmd,
    shell=True,
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace'
)

if result.stdout:
    print(result.stdout)
if result.stderr:
    print(result.stderr)

if result.returncode != 0:
    print("❌ Failed to add openblas-src")
    print("\n💡 Make sure Rust/Cargo is installed:")
    print("   https://rustup.rs/")
    sys.exit(1)

print("✅ Added openblas-src with static feature")

# Step 2: Clean previous build artifacts
print("\n🧹 Step 2: Cleaning previous build artifacts...")
target_dir = concept_mesh_dir / "target"
if target_dir.exists():
    print("Cleaning target directory...")
    import shutil
    shutil.rmtree(target_dir, ignore_errors=True)
    print("✅ Cleaned")

# Step 3: Build with maturin
print("\n📦 Step 3: Building concept_mesh_rs wheel...")

build_cmd = f'"{sys.executable}" -m maturin build --release'
print(f"Running: {build_cmd}")

result = subprocess.run(
    build_cmd,
    shell=True,
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace',
    env=env
)

# Print output
if result.stdout:
    print("BUILD OUTPUT:")
    print(result.stdout)
if result.stderr:
    print("BUILD ERRORS/WARNINGS:")
    print(result.stderr)

if result.returncode != 0:
    print("\n❌ Build failed!")
    
    if "Microsoft Visual C++" in str(result.stderr):
        print("\n💡 Visual C++ Build Tools required!")
        print("   Download and install:")
        print("   https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022")
        print("   Select 'Desktop development with C++' workload")
    elif "linker `link.exe` not found" in str(result.stderr):
        print("\n💡 Run from Visual Studio Developer Command Prompt")
        print("   Or ensure VS Build Tools are in PATH")
    
    sys.exit(1)

print("\n✅ Build successful!")

# Step 4: Install the wheel
wheel_dir = concept_mesh_dir / "target" / "wheels"
wheels = list(wheel_dir.glob("concept_mesh_rs-*.whl"))

if not wheels:
    print("❌ No wheel found in target/wheels/")
    print("   Contents of target/wheels/:")
    if wheel_dir.exists():
        for item in wheel_dir.iterdir():
            print(f"   - {item.name}")
    sys.exit(1)

wheel_path = wheels[0]
print(f"\n📦 Step 4: Installing wheel: {wheel_path.name}")

# Force reinstall to ensure it goes to site-packages
install_cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheel_path)]
print(f"Running: {' '.join(install_cmd)}")

install_result = subprocess.run(
    install_cmd,
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace'
)

print(install_result.stdout)
if install_result.stderr:
    print(install_result.stderr)

if install_result.returncode != 0:
    print("❌ Install failed!")
    sys.exit(1)

print("✅ Wheel installed successfully!")

# Step 5: Verify installation
print("\n🧪 Step 5: Verifying installation...")
os.chdir(kha_path)

# Check in site-packages
site_packages = Path(sys.executable).parent.parent / "Lib" / "site-packages"
pyd_files = list(site_packages.glob("concept_mesh_rs*.pyd"))
if pyd_files:
    print(f"✅ Found in site-packages: {pyd_files[0].name}")
else:
    print("⚠️ Not found in site-packages, checking other locations...")

# Test import
test_cmd = [sys.executable, "-c", "import concept_mesh_rs; print(f'✅ Imported from: {concept_mesh_rs.__file__}')"]
test_result = subprocess.run(
    test_cmd,
    capture_output=True,
    text=True,
    encoding='utf-8'
)

print(test_result.stdout)
if test_result.stderr:
    print(test_result.stderr)

if test_result.returncode != 0:
    print("❌ Import test failed!")
    print("\n💡 Try manually:")
    print(f"   {sys.executable}")
    print("   >>> import concept_mesh_rs")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ PENROSE CONCEPT MESH BUILT AND INSTALLED!")
print("\n🚀 Next steps:")
print("   1. Run the MCP fix: python fix_penrose_mcp_complete.py")
print("   2. Restart server: python enhanced_launcher.py")
print("   3. Verify no 'mock' warnings in logs")
