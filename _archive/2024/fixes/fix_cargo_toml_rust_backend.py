#!/usr/bin/env python3
"""
Fix Cargo.toml to use pure Rust backend (no OpenBLAS)
"""

import sys
from pathlib import Path

print("🔧 FIXING CARGO.TOML FOR PURE RUST BACKEND")
print("=" * 60)

cargo_toml_path = Path("concept_mesh/Cargo.toml")

if not cargo_toml_path.exists():
    print("❌ Cargo.toml not found!")
    sys.exit(1)

# Read the current content
content = cargo_toml_path.read_text()

# Backup
backup_path = cargo_toml_path.with_suffix('.toml.backup_openblas')
backup_path.write_text(content)
print(f"✅ Created backup: {backup_path}")

# Fix 1: Change ndarray-linalg to use rust backend
print("\n📝 Fixing ndarray-linalg to use pure Rust...")
content = content.replace(
    'ndarray-linalg = { version = "0.16", features = ["openblas-system"] }',
    'ndarray-linalg = { version = "0.16", default-features = false, features = ["rust"] }'
)

# Fix 2: Remove or comment out openblas-src
print("📝 Removing openblas-src dependency...")
lines = content.split('\n')
new_lines = []
for line in lines:
    if line.strip().startswith('openblas-src ='):
        new_lines.append('# ' + line + '  # Commented out - using pure Rust')
    else:
        new_lines.append(line)

content = '\n'.join(new_lines)

# Write the fixed content
cargo_toml_path.write_text(content)
print("✅ Updated Cargo.toml to use pure Rust backend")

print("\n📋 Current ndarray-linalg configuration:")
for line in content.split('\n'):
    if 'ndarray-linalg' in line:
        print(f"   {line.strip()}")

print("\n✅ Done! Now you can build without OpenBLAS:")
print("   cd concept_mesh")
print("   maturin build --release")
print("   pip install target/wheels/concept_mesh_rs-*.whl")
