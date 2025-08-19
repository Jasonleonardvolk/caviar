#!/usr/bin/env python3
"""
TORI Type Shadowing Fix Script

PURPOSE:
    Fixes the types.py shadowing issue that prevents proper Python module imports.
    This occurs when a local types.py file shadows Python's built-in types module,
    causing import conflicts during Rust extension builds.

WHAT IT DOES:
    1. Renames concept_mesh/types.py to concept_mesh/mesh_types.py
    2. Updates all import statements in affected files
    3. Installs maturin for Rust wheel building
    4. Attempts to build the concept_mesh_rs wheel
    5. Tests the installation

USAGE:
    python fix_types_shadowing.py

FILES MODIFIED:
    - concept_mesh/types.py -> concept_mesh/mesh_types.py
    - concept_mesh/interface.py (import updates)
    - concept_mesh/__init__.py (import updates) 
    - concept_mesh/loader.py (import updates)

AUTHOR: TORI System Maintenance
LAST UPDATED: 2025-01-26
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Constants for maintainability
KHA_ROOT = Path(__file__).parent.absolute()
CONCEPT_MESH_DIR = KHA_ROOT / "concept_mesh"
TYPES_FILE = CONCEPT_MESH_DIR / "types.py"
NEW_TYPES_FILE = CONCEPT_MESH_DIR / "mesh_types.py"

# Files that may import from types.py
FILES_TO_UPDATE = [
    CONCEPT_MESH_DIR / "interface.py",
    CONCEPT_MESH_DIR / "__init__.py",
    CONCEPT_MESH_DIR / "loader.py",
]

def print_header(message: str) -> None:
    """Print a formatted header message"""
    print(f"\n{'=' * 60}")
    print(f"{message:^60}")
    print(f"{'=' * 60}\n")

def print_step(step: str, message: str) -> None:
    """Print a step message with formatting"""
    print(f"\n{step} {message}")

def fix_types_shadowing() -> bool:
    """Fix the types.py shadowing issue"""
    print_header("FIXING TYPES.PY SHADOWING ISSUE")
    print(f"Working directory: {KHA_ROOT}")
    
    # Step 1: Check if types.py exists and rename it
    if TYPES_FILE.exists():
        print_step("📝", f"Renaming {TYPES_FILE} -> {NEW_TYPES_FILE}")
        try:
            shutil.move(str(TYPES_FILE), str(NEW_TYPES_FILE))
            print("✅ Renamed to avoid shadowing Python's builtin types module")
        except Exception as e:
            print(f"❌ Failed to rename types.py: {e}")
            return False
        
        # Step 2: Update imports in other files
        print_step("📝", "Updating imports in other files...")
        updated_files = []
        
        for file_path in FILES_TO_UPDATE:
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    
                    # Replace various import patterns
                    old_patterns = [
                        "from .types import",
                        "from types import",
                        "import types"
                    ]
                    new_patterns = [
                        "from .mesh_types import",
                        "from .mesh_types import", 
                        "import .mesh_types as types"
                    ]
                    
                    content_modified = False
                    for old_pattern, new_pattern in zip(old_patterns, new_patterns):
                        if old_pattern in content:
                            content = content.replace(old_pattern, new_pattern)
                            content_modified = True
                    
                    if content_modified:
                        file_path.write_text(content, encoding='utf-8')
                        updated_files.append(file_path.name)
                        print(f"  ✅ Updated imports in {file_path.name}")
                        
                except Exception as e:
                    print(f"  ❌ Failed to update {file_path.name}: {e}")
                    return False
        
        if updated_files:
            print(f"\n✅ Updated imports in {len(updated_files)} files: {', '.join(updated_files)}")
        else:
            print("\nℹ️ No import updates needed")
    else:
        print("ℹ️ types.py not found - may already be fixed")
    
    return True

def install_maturin() -> bool:
    """Install or upgrade maturin for Rust wheel building"""
    print_step("📦", "Installing/upgrading maturin...")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-U", "maturin"], 
            capture_output=True, 
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ Maturin installed successfully!")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()[:200]}...")  # Truncate long output
            return True
        else:
            print(f"❌ Maturin installation failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Maturin installation timed out")
        return False
    except Exception as e:
        print(f"❌ Unexpected error installing maturin: {e}")
        return False

def build_concept_mesh_wheel() -> bool:
    """Build the concept_mesh_rs wheel using maturin"""
    print_step("📦", "Building concept_mesh_rs wheel...")
    
    if not CONCEPT_MESH_DIR.exists():
        print(f"❌ Concept mesh directory not found: {CONCEPT_MESH_DIR}")
        return False
    
    # Change to concept_mesh directory
    original_cwd = os.getcwd()
    try:
        os.chdir(CONCEPT_MESH_DIR)
        
        # Build the wheel
        result = subprocess.run(
            [sys.executable, "-m", "maturin", "build", "--release"], 
            capture_output=True, 
            text=True,
            timeout=600  # 10 minute timeout for build
        )
        
        if result.returncode == 0:
            print("✅ Build successful!")
            
            # Try to install the wheel
            return install_wheel()
        else:
            print(f"❌ Build failed: {result.stderr}")
            
            # Check for common issues and provide suggestions
            if "openblas" in result.stderr.lower() or "lapack" in result.stderr.lower():
                print("\n💡 OpenBLAS/LAPACK issue detected!")
                print("   Try these commands:")
                print("   cd concept_mesh")
                print("   cargo add openblas-src --features=static")
                print("   Then run this script again")
            
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Build timed out (>10 minutes)")
        return False
    except Exception as e:
        print(f"❌ Unexpected build error: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def install_wheel() -> bool:
    """Install the built wheel and test import"""
    wheel_dir = CONCEPT_MESH_DIR / "target" / "wheels"
    
    if not wheel_dir.exists():
        print(f"❌ Wheel directory not found: {wheel_dir}")
        return False
    
    # Find the most recent wheel
    wheels = list(wheel_dir.glob("concept_mesh_rs-*.whl"))
    if not wheels:
        print("❌ No wheel found after build")
        return False
    
    # Sort by modification time, get the newest
    wheel_path = max(wheels, key=lambda p: p.stat().st_mtime)
    
    print(f"\n📦 Installing wheel: {wheel_path.name}")
    
    try:
        # Install the wheel
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", str(wheel_path), "--force-reinstall"], 
            capture_output=True, 
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ Wheel installed successfully!")
            
            # Test the import
            return test_import()
        else:
            print(f"❌ Install failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Installation timed out")
        return False
    except Exception as e:
        print(f"❌ Unexpected installation error: {e}")
        return False

def test_import() -> bool:
    """Test importing the installed module"""
    print_step("🧪", "Testing import...")
    
    try:
        test_result = subprocess.run(
            [sys.executable, "-c", "import concept_mesh_rs; print(f'✅ Imported from: {concept_mesh_rs.__file__}')"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if test_result.returncode == 0:
            print(test_result.stdout.strip())
            return True
        else:
            print(f"❌ Import test failed: {test_result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Import test timed out")
        return False
    except Exception as e:
        print(f"❌ Import test error: {e}")
        return False

def main() -> int:
    """Main entry point"""
    try:
        # Step 1: Fix types shadowing
        if not fix_types_shadowing():
            return 1
        
        # Step 2: Install maturin
        if not install_maturin():
            return 1
        
        # Step 3: Build and install wheel
        if not build_concept_mesh_wheel():
            return 1
        
        # Success!
        print_header("✅ TYPES SHADOWING FIXED!")
        print("\n🚀 Next steps:")
        print("   1. Run: python fix_penrose_mcp_complete.py")
        print("   2. The import should work now that types.py is renamed")
        print("   3. Test your concept mesh functionality")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Script interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)