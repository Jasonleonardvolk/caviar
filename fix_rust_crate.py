#!/usr/bin/env python3
"""Quick Rust Crate Fixes"""
import os
import subprocess

def fix_rust_crate():
    print("\nðŸ¦€ Applying Rust crate fixes...")
    
    # Add missing dependencies
    deps = ["num-complex", "colored", "once_cell", "serde_json"]
    for dep in deps:
        print(f"  Adding {dep}...")
        subprocess.run(["cargo", "add", dep], capture_output=True)
    
    # Create missing module files
    missing_mods = ["src/ingest/mod.rs", "src/wal/mod.rs"]
    for mod_file in missing_mods:
        if not os.path.exists(mod_file):
            os.makedirs(os.path.dirname(mod_file), exist_ok=True)
            with open(mod_file, 'w') as f:
                f.write(f'//! {os.path.basename(os.path.dirname(mod_file))} module\n')
            print(f"  Created {mod_file}")
    
    print("\nâœ… Basic Rust fixes applied!")
    print("Run 'cargo check' to see remaining issues")

if __name__ == "__main__":
    fix_rust_crate()
