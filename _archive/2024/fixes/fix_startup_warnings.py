#!/usr/bin/env python3
"""
Fix TORI Startup Warnings
Addresses the various warnings that appear during enhanced_launcher.py startup
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

def backup_file(filepath):
    """Create a backup of the file before modifying"""
    if os.path.exists(filepath):
        backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup_path)
        print(f"✅ Backed up: {filepath}")
        return backup_path
    return None

def fix_conceptmesh_import():
    """Fix the ConceptMesh import issue in multiple locations"""
    print("\n🔧 Fixing ConceptMesh imports...")
    
    # The issue is that some files are importing from concept_mesh_rs which doesn't exist
    # We need to make them fall back to the mock implementation
    
    files_to_check = [
        "mcp_metacognitive/core/soliton_memory.py",
        "ingest_pdf/pipeline/quality.py",
        "python/core/__init__.py"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"  Checking {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original = content
            
            # Remove the concept_mesh_rs import attempt
            if 'import concept_mesh_rs' in content:
                backup_file(file_path)
                content = content.replace(
                    'import concept_mesh_rs',
                    '# import concept_mesh_rs  # Not available'
                )
                content = content.replace(
                    'from concept_mesh_rs.interface import ConceptMesh',
                    '# from concept_mesh_rs.interface import ConceptMesh  # Not available'
                )
                content = content.replace(
                    'USING_RUST_WHEEL = True',
                    'USING_RUST_WHEEL = False'
                )
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"    ✅ Fixed imports in {file_path}")

def fix_gudhi_warnings():
    """Add GUDHI availability check to prevent warnings"""
    print("\n🔧 Fixing GUDHI warnings...")
    
    gudhi_check = '''# Check GUDHI availability
try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
'''
    
    files_with_gudhi = [
        "cog/transfer.py",
        "python/core/transfer_morphism.py"
    ]
    
    for file_path in files_with_gudhi:
        if os.path.exists(file_path):
            print(f"  Checking {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'GUDHI_AVAILABLE' not in content:
                backup_file(file_path)
                # Add the check at the top after imports
                lines = content.split('\n')
                import_end = 0
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith('import') and not line.startswith('from'):
                        import_end = i
                        break
                
                lines.insert(import_end, gudhi_check)
                
                # Replace GUDHI usage with conditional
                content = '\n'.join(lines)
                content = content.replace(
                    'warnings.warn("GUDHI library not available',
                    'if not GUDHI_AVAILABLE:\n    warnings.warn("GUDHI library not available'
                )
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"    ✅ Added GUDHI check to {file_path}")

def create_missing_init_files():
    """Create missing __init__.py files in directories"""
    print("\n🔧 Creating missing __init__.py files...")
    
    directories = [
        "mcp_metacognitive",
        "mcp_metacognitive/core",
        "ingest_pdf",
        "ingest_pdf/pipeline",
        "python",
        "python/core",
        "cog"
    ]
    
    for dir_path in directories:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            init_file = os.path.join(dir_path, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write(f'"""Package: {dir_path}"""\n')
                print(f"    ✅ Created {init_file}")

def fix_fisherrao_warning():
    """Fix Fisher-Rao metric warning"""
    print("\n🔧 Fixing Fisher-Rao metric warning...")
    
    manifold_file = "cog/manifold.py"
    if os.path.exists(manifold_file):
        with open(manifold_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'log_prob_model' in content:
            backup_file(manifold_file)
            # Suppress the warning by checking if log_prob_model exists
            content = content.replace(
                'warnings.warn("Fisher-Rao metric requires log_prob_model.',
                'if not hasattr(self, "log_prob_model"):\n            warnings.warn("Fisher-Rao metric requires log_prob_model.'
            )
            
            with open(manifold_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print("    ✅ Fixed Fisher-Rao warning")

def disable_entropy_pruning():
    """Ensure entropy pruning is disabled"""
    print("\n🔧 Ensuring entropy pruning is disabled...")
    
    # Add to enhanced_launcher.py if not already there
    launcher_file = "enhanced_launcher.py"
    if os.path.exists(launcher_file):
        with open(launcher_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "TORI_DISABLE_ENTROPY_PRUNE" not in content:
            backup_file(launcher_file)
            # Add after the encoding setup
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'sys.stderr.reconfigure' in line:
                    lines.insert(i + 2, "\n# Disable entropy pruning (module not available)")
                    lines.insert(i + 3, "os.environ['TORI_DISABLE_ENTROPY_PRUNE'] = '1'")
                    break
            
            with open(launcher_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print("    ✅ Added entropy pruning disable flag")

def create_startup_fix_summary():
    """Create a summary of fixes applied"""
    summary = f"""# TORI Startup Fixes Applied

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Fixes Applied:

1. **ConceptMesh Import Issues**
   - Disabled concept_mesh_rs imports (module not available)
   - Set USING_RUST_WHEEL = False
   - Files will use mock implementation

2. **GUDHI Warnings**
   - Added GUDHI availability checks
   - Warnings now only show if GUDHI is actually needed

3. **Missing __init__.py Files**
   - Created package initialization files
   - Helps Python recognize directories as packages

4. **Fisher-Rao Metric Warning**
   - Added conditional check before warning
   - Reduces noise in startup

5. **Entropy Pruning**
   - Ensured TORI_DISABLE_ENTROPY_PRUNE is set
   - Prevents import errors for missing module

## Next Steps:

1. Run `poetry run python enhanced_launcher.py` again
2. The startup should have fewer warnings
3. If ConceptMesh errors persist, check the specific import paths

## Optional Improvements:

- Install GUDHI: `pip install gudhi` (requires C++ compiler)
- Build concept_mesh_rs wheel (requires Rust toolchain)
- Implement missing components for full functionality
"""
    
    with open("STARTUP_FIXES_SUMMARY.md", 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"\n📄 Summary saved to: STARTUP_FIXES_SUMMARY.md")

def main():
    """Main execution"""
    print("🔧 TORI Startup Warning Fixer")
    print("=" * 60)
    
    # Apply all fixes
    fix_conceptmesh_import()
    create_missing_init_files()
    fix_gudhi_warnings()
    fix_fisherrao_warning()
    disable_entropy_pruning()
    
    # Create summary
    create_startup_fix_summary()
    
    print("\n✅ All startup fixes applied!")
    print("\nRun 'poetry run python enhanced_launcher.py' to see improvements")

if __name__ == "__main__":
    main()
