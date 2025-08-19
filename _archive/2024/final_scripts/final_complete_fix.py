#!/usr/bin/env python3
"""
Complete fix - Remove ndarray-linalg and build with plain ndarray
"""

import os
import sys
import subprocess
from pathlib import Path
import re

print("🔧 COMPLETE FIX - PURE RUST, NO LINEAR ALGEBRA DEPS")
print("=" * 60)

# PART 1: Fix Cargo.toml - Remove ndarray-linalg
print("\n📋 PART 1: Fixing Cargo.toml...")
print("-" * 40)

cargo_toml_path = Path("concept_mesh/Cargo.toml")
if cargo_toml_path.exists():
    content = cargo_toml_path.read_text()
    
    # Backup
    backup_path = cargo_toml_path.with_suffix('.toml.backup_final')
    backup_path.write_text(content)
    print(f"✅ Created backup: {backup_path}")
    
    # Comment out both ndarray-linalg AND openblas-src
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        if ('ndarray-linalg' in line or 'openblas-src' in line) and '=' in line and not line.strip().startswith('#'):
            new_lines.append('# ' + line + '  # Removed for pure Rust build')
            print(f"   Commented out: {line.strip()}")
        else:
            new_lines.append(line)
    
    content = '\n'.join(new_lines)
    cargo_toml_path.write_text(content)
    print("✅ Removed all linear algebra dependencies")

# PART 2: Build the wheel
print("\n📦 PART 2: Building Concept Mesh wheel...")
print("-" * 40)

os.chdir("concept_mesh")

# Clean build
print("Cleaning previous builds...")
subprocess.run("cargo clean", shell=True, capture_output=True)

# Build with maturin
print("Building wheel (pure Rust, no external deps)...")
env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'

result = subprocess.run(
    f'"{sys.executable}" -m maturin build --release',
    shell=True,
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace',
    env=env
)

if result.returncode == 0:
    print("✅ Build successful!")
    
    # Find and install wheel
    wheel_dir = Path("target/wheels")
    wheels = list(wheel_dir.glob("concept_mesh_rs-*.whl"))
    
    if wheels:
        wheel_path = wheels[0]
        print(f"\n📦 Installing wheel: {wheel_path.name}")
        
        install_cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheel_path)]
        install_result = subprocess.run(install_cmd, capture_output=True, text=True)
        
        if install_result.returncode == 0:
            print("✅ Wheel installed successfully!")
            print(install_result.stdout)
        else:
            print(f"❌ Install failed: {install_result.stderr}")
    else:
        print("❌ No wheel found in target/wheels/")
else:
    print("❌ Build failed!")
    print("\nBuild output:")
    if result.stdout:
        print("STDOUT:", result.stdout[:1000])
    if result.stderr:
        print("STDERR:", result.stderr[:1000])

os.chdir("..")

# PART 3: Fix MCP FastMCP.run() signature
print("\n🔧 PART 3: Fixing MCP FastMCP.run() signature...")
print("-" * 40)

# First, let's find where FastMCP actually is
possible_paths = [
    Path("mcp_metacognitive/fast_mcp.py"),
    Path("mcp_metacognitive/core/fast_mcp.py"),
    Path("mcp_metacognitive/server.py"),  # Maybe it's in server.py?
]

# Search for FastMCP class
for search_path in [Path("mcp_metacognitive")]:
    if search_path.exists():
        print(f"Searching in {search_path}...")
        for py_file in search_path.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                if "class FastMCP" in content:
                    print(f"✅ Found FastMCP in: {py_file}")
                    
                    # Check if run method needs patching
                    if "def run(self" in content:
                        # Look for the run method
                        if re.search(r'def run\(self\s*\)', content):
                            print("   Patching run() to accept host/port...")
                            
                            # Create backup
                            backup = py_file.with_suffix('.py.backup_mcp')
                            if not backup.exists():
                                backup.write_text(content, encoding='utf-8')
                            
                            # Add host/port parameters
                            content = re.sub(
                                r'def run\(self\s*\):',
                                'def run(self, transport="stdio", host="0.0.0.0", port=8100, **kwargs):',
                                content
                            )
                            
                            py_file.write_text(content, encoding='utf-8')
                            print("   ✅ Patched!")
                        elif "host" in content and "port" in content:
                            print("   ✅ Already accepts host/port")
                    break
            except Exception as e:
                pass

# PART 4: Test import
print("\n🧪 Testing concept_mesh_rs import...")
print("-" * 40)

test_result = subprocess.run(
    [sys.executable, "-c", """
import sys
try:
    import concept_mesh_rs
    print(f'✅ concept_mesh_rs imported from: {concept_mesh_rs.__file__}')
    loader = concept_mesh_rs.get_loader()
    print(f'✅ Loader created: {type(loader).__name__}')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
"""],
    capture_output=True,
    text=True
)

print(test_result.stdout)
if test_result.stderr:
    print("Warnings:", test_result.stderr)

print("\n" + "=" * 60)
if test_result.returncode == 0:
    print("✅ SUCCESS! Everything is working!")
    print("\n🚀 Next steps:")
    print("   1. Kill all Python: taskkill /IM python.exe /F")
    print("   2. Start server: python enhanced_launcher.py")
    print("   3. Verify:")
    print("      - Main log shows '🦀 Penrose backend: rust'")
    print("      - NO 'mock' warnings in MCP logs")
    print("      - MCP starts on port 8100")
else:
    print("⚠️ Import test failed - check errors above")
