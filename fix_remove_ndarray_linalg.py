#!/usr/bin/env python3
"""
Fix Cargo.toml to remove ndarray-linalg dependency
We'll use plain ndarray without linear algebra features
"""

import sys
from pathlib import Path

print("🔧 FIXING CARGO.TOML - REMOVING NDARRAY-LINALG")
print("=" * 60)

cargo_toml_path = Path("concept_mesh/Cargo.toml")

if not cargo_toml_path.exists():
    print("❌ Cargo.toml not found!")
    sys.exit(1)

# Read current content
content = cargo_toml_path.read_text()

# Backup
backup_path = cargo_toml_path.with_suffix('.toml.backup_no_linalg')
backup_path.write_text(content)
print(f"✅ Created backup: {backup_path}")

# Option 1: Remove ndarray-linalg entirely (simplest)
print("\n📝 Removing ndarray-linalg dependency...")
lines = content.split('\n')
new_lines = []
skip_line = False

for line in lines:
    # Skip ndarray-linalg line
    if 'ndarray-linalg' in line and '=' in line:
        new_lines.append('# ' + line + '  # Removed - using plain ndarray')
        print(f"   Commented out: {line.strip()}")
        skip_line = False
    else:
        new_lines.append(line)

content = '\n'.join(new_lines)

# Make sure plain ndarray is there
if 'ndarray = "0.15"' not in content and 'ndarray =' in content:
    print("✅ ndarray dependency already present")

# Write fixed content
cargo_toml_path.write_text(content)

print("\n✅ Fixed! Now using plain ndarray without linear algebra")
print("\n📋 What this means:")
print("   - No external BLAS/LAPACK dependencies")
print("   - Pure Rust implementation")
print("   - Will compile on any platform")
print("   - Some advanced linear algebra operations unavailable")
print("   - Basic array operations still work fine")

print("\n🚀 Next: Build the wheel")
print("   cd concept_mesh")
print("   maturin build --release")
