#!/usr/bin/env python3
"""
Quick ConceptMesh Import Diagnostic
Finds exactly which files are causing import errors
"""

import os
import sys
import importlib
import traceback
from pathlib import Path

def check_import_chain():
    """Check the import chain for concept_mesh_rs"""
    print("🔍 Checking ConceptMesh import chain...\n")
    
    # First, check if concept_mesh_rs exists and what it contains
    print("1. Checking concept_mesh_rs package:")
    if os.path.exists("concept_mesh_rs"):
        print("  ✅ concept_mesh_rs directory exists")
        
        # Check __init__.py
        init_file = "concept_mesh_rs/__init__.py"
        if os.path.exists(init_file):
            print(f"  ✅ {init_file} exists")
            with open(init_file, 'r') as f:
                content = f.read()
                if 'ConceptMeshLoader' in content:
                    print("  ✅ ConceptMeshLoader is exported")
                else:
                    print("  ❌ ConceptMeshLoader NOT found in exports")
        else:
            print(f"  ❌ {init_file} missing")
    else:
        print("  ❌ concept_mesh_rs directory missing")
    
    print("\n2. Testing imports:")
    
    # Test various import patterns
    test_imports = [
        ("from concept_mesh_rs import ConceptMesh", "ConceptMesh"),
        ("from concept_mesh_rs import ConceptMeshLoader", "ConceptMeshLoader"),
        ("from concept_mesh_rs.interface import ConceptMesh", "ConceptMesh from interface"),
        ("from concept_mesh_rs.loader import ConceptMeshLoader", "ConceptMeshLoader from loader"),
        ("import concept_mesh_rs", "concept_mesh_rs module")
    ]
    
    for import_stmt, desc in test_imports:
        try:
            exec(import_stmt)
            print(f"  ✅ {desc}: OK")
        except Exception as e:
            print(f"  ❌ {desc}: {type(e).__name__}: {str(e)}")

def find_import_locations():
    """Find all files trying to import concept_mesh_rs"""
    print("\n3. Files attempting to import concept_mesh_rs:")
    
    import_files = []
    
    # Search in key directories
    search_dirs = ['mcp_metacognitive', 'python', 'ingest_pdf', 'api', 'core']
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        for root, dirs, files in os.walk(search_dir):
            dirs[:] = [d for d in dirs if d not in {'__pycache__', '.git'}]
            
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if 'concept_mesh_rs' in content and 'class ConceptMesh' not in content:
                            import_files.append(filepath)
                            
                            # Show the specific import line
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if 'concept_mesh_rs' in line and ('import' in line or 'from' in line):
                                    print(f"\n  📄 {filepath}:{i+1}")
                                    print(f"     {line.strip()}")
                    except:
                        pass
    
    return import_files

def suggest_fix():
    """Suggest specific fixes"""
    print("\n4. Suggested fixes:")
    
    if not os.path.exists("concept_mesh_rs/__init__.py"):
        print("\n  🔧 Run: python robust_tori_fix.py")
        print("     This will create the complete stub module")
    else:
        # Check if ConceptMeshLoader is missing
        with open("concept_mesh_rs/__init__.py", 'r') as f:
            content = f.read()
        
        if 'ConceptMeshLoader' not in content:
            print("\n  🔧 ConceptMeshLoader is missing from exports")
            print("     Add to concept_mesh_rs/__init__.py:")
            print("     from .loader import ConceptMeshLoader")
            print("     __all__ = [..., 'ConceptMeshLoader']")

def main():
    print("=" * 60)
    print("ConceptMesh Import Diagnostic")
    print("=" * 60)
    
    check_import_chain()
    import_files = find_import_locations()
    suggest_fix()
    
    print("\n" + "=" * 60)
    print(f"Summary: Found {len(import_files)} files with concept_mesh_rs imports")
    
    if import_files:
        print("\nTo fix all at once, run:")
        print("  python robust_tori_fix.py")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
