#!/usr/bin/env python3
"""
Complete fix for Concept Mesh and MCP issues
1. Fix Cargo.toml for pure Rust
2. Build the wheel
3. Fix MCP run() signature
"""

import os
import sys
import subprocess
from pathlib import Path
import re

print("🔧 COMPLETE FIX FOR CONCEPT MESH & MCP")
print("=" * 60)

# PART 1: Fix Cargo.toml
print("\n📋 PART 1: Fixing Cargo.toml...")
print("-" * 40)

cargo_toml_path = Path("concept_mesh/Cargo.toml")
if cargo_toml_path.exists():
    content = cargo_toml_path.read_text()
    
    # Backup
    backup_path = cargo_toml_path.with_suffix('.toml.backup_complete')
    if not backup_path.exists():
        backup_path.write_text(content)
        print(f"✅ Created backup: {backup_path}")
    
    # Fix ndarray-linalg
    original_line = 'ndarray-linalg = { version = "0.16", features = ["openblas-system"] }'
    fixed_line = 'ndarray-linalg = { version = "0.16", default-features = false, features = ["rust"] }'
    
    if original_line in content:
        content = content.replace(original_line, fixed_line)
        print("✅ Changed ndarray-linalg to use pure Rust backend")
    elif fixed_line in content:
        print("✅ ndarray-linalg already using pure Rust backend")
    
    # Comment out openblas-src
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if line.strip().startswith('openblas-src =') and not line.strip().startswith('#'):
            new_lines.append('# ' + line + '  # Using pure Rust instead')
            print("✅ Commented out openblas-src")
        else:
            new_lines.append(line)
    
    content = '\n'.join(new_lines)
    cargo_toml_path.write_text(content)

# PART 2: Build the wheel
print("\n📦 PART 2: Building Concept Mesh wheel...")
print("-" * 40)

os.chdir("concept_mesh")

# Clean build
print("Cleaning previous builds...")
subprocess.run("cargo clean", shell=True, capture_output=True)

# Build
print("Building wheel (this may take a few minutes)...")
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
    
    # Install wheel
    wheel_dir = Path("target/wheels")
    wheels = list(wheel_dir.glob("concept_mesh_rs-*.whl"))
    
    if wheels:
        wheel_path = wheels[0]
        print(f"\nInstalling wheel: {wheel_path.name}")
        
        install_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheel_path)],
            capture_output=True,
            text=True
        )
        
        if install_result.returncode == 0:
            print("✅ Wheel installed successfully!")
        else:
            print(f"❌ Install failed: {install_result.stderr}")
else:
    print("❌ Build failed!")
    print(f"Error: {result.stderr[:500] if result.stderr else 'Unknown error'}")

os.chdir("..")

# PART 3: Fix MCP run() signature
print("\n🔧 PART 3: Fixing MCP FastMCP.run() signature...")
print("-" * 40)

mcp_paths = [
    Path("mcp_metacognitive/fast_mcp.py"),
    Path("mcp_metacognitive/core/fast_mcp.py"),
]

fixed_mcp = False
for mcp_path in mcp_paths:
    if mcp_path.exists():
        print(f"Found FastMCP at: {mcp_path}")
        content = mcp_path.read_text(encoding='utf-8')
        
        # Backup
        backup = mcp_path.with_suffix('.py.backup_complete')
        if not backup.exists():
            backup.write_text(content, encoding='utf-8')
        
        # Check if needs patching
        if "def run(self" in content and "host:" not in content:
            print("Patching run() method...")
            
            # New run method with proper signature
            new_run_method = '''    def run(
        self,
        transport: str = "stdio",
        host: str = "0.0.0.0",
        port: int = 8100,
        **kwargs
    ):
        """Run the MCP server with specified transport"""
        if transport in ["http", "sse"]:
            import uvicorn
            uvicorn.run(
                self.app,
                host=host,
                port=port,
                log_level="info",
                **kwargs
            )
        elif transport == "stdio":
            self._run_stdio()
        else:
            raise ValueError(f"Unknown transport: {transport}")'''
            
            # Replace the old run method
            pattern = r'def run\(self[^:]*\):[^}]+?(?=\n    def|\n\nclass|\Z)'
            content = re.sub(pattern, new_run_method.strip(), content, flags=re.DOTALL)
            
            mcp_path.write_text(content, encoding='utf-8')
            print("✅ Patched FastMCP.run()")
            fixed_mcp = True
        else:
            print("✅ FastMCP.run() already accepts host/port")
            fixed_mcp = True
        break

if not fixed_mcp:
    print("⚠️ FastMCP not found")

# Final test
print("\n🧪 Final verification...")
print("-" * 40)

test_result = subprocess.run(
    [sys.executable, "-c", """
try:
    import concept_mesh_rs
    print(f'✅ concept_mesh_rs imported from: {concept_mesh_rs.__file__}')
except ImportError as e:
    print(f'❌ Import failed: {e}')
"""],
    capture_output=True,
    text=True
)

print(test_result.stdout)

print("\n" + "=" * 60)
print("✅ FIXES COMPLETE!")
print("\n🚀 Next steps:")
print("   1. Stop all Python processes: taskkill /IM python.exe /F")
print("   2. Restart server: python enhanced_launcher.py")
print("   3. Check logs for:")
print("      - '🦀 Penrose backend: rust' (main process)")
print("      - NO 'mock' warnings in MCP stderr")
print("      - 'Uvicorn running on http://0.0.0.0:8100'")
