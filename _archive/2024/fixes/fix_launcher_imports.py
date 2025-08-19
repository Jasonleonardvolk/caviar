#!/usr/bin/env python3
"""
Enhanced Launcher Fix for Cognitive Interface and Concept Mesh
=============================================================

This module patches the enhanced_launcher.py to properly handle
the cognitive_interface and concept_mesh imports.

To use:
1. Run this script once to patch your launcher
2. Then run enhanced_launcher.py as usual
"""

import os
import sys
from pathlib import Path

def patch_enhanced_launcher():
    """Patch the enhanced launcher to fix import issues"""
    
    script_dir = Path(__file__).parent.absolute()
    launcher_path = script_dir / "enhanced_launcher.py"
    
    if not launcher_path.exists():
        print("❌ enhanced_launcher.py not found!")
        return False
    
    print("🔧 Patching enhanced_launcher.py to fix imports...")
    
    # Read the current launcher
    with open(launcher_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already patched
    if "# COGNITIVE_INTERFACE_PATCH" in content:
        print("✅ Launcher already patched!")
        return True
    
    # Find the imports section (after the UTF-8 setup)
    insert_position = content.find("# Optional MCP bridge import")
    
    if insert_position == -1:
        print("❌ Could not find insertion point in launcher!")
        return False
    
    # Create the patch
    patch = '''
# COGNITIVE_INTERFACE_PATCH - Fix for cognitive_interface and concept_mesh imports
# Add project root to Python path for proper imports
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Ensure ingest_pdf is importable
ingest_pdf_path = project_root / "ingest_pdf"
if ingest_pdf_path.exists() and str(ingest_pdf_path) not in sys.path:
    sys.path.insert(0, str(ingest_pdf_path.parent))

# Try to import cognitive_interface
try:
    from ingest_pdf.cognitive_interface import add_concept_diff
    COGNITIVE_INTERFACE_AVAILABLE = True
except ImportError:
    COGNITIVE_INTERFACE_AVAILABLE = False
    add_concept_diff = None

# Setup concept_mesh - either real or mock
try:
    from python.core import ConceptMeshConnector
    CONCEPT_MESH_AVAILABLE = True
except ImportError:
    # Create mock concept_mesh if not available
    concept_mesh_dir = project_root / "concept_mesh"
    if not concept_mesh_dir.exists():
        concept_mesh_dir.mkdir(exist_ok=True)
        mock_init = concept_mesh_dir / "__init__.py"
        mock_init.write_text("""
# Mock ConceptMeshConnector for testing
class ConceptMeshConnector:
    def __init__(self, url=None):
        self.url = url or "http://localhost:8003/api/mesh"
    def connect(self):
        return True
    def get_concepts(self):
        return []
    def add_concept(self, concept):
        return {"id": "mock_id", "concept": concept}
""")
    
    # Try importing again
    try:
        from python.core import ConceptMeshConnector
        CONCEPT_MESH_AVAILABLE = True
    except ImportError:
        CONCEPT_MESH_AVAILABLE = False
        ConceptMeshConnector = None

'''
    
    # Insert the patch
    new_content = content[:insert_position] + patch + content[insert_position:]
    
    # Also fix the __init__.py creation in fix_concept_mesh_data method
    # Find the ConceptMeshDataFixer class
    fixer_position = content.find("class ConceptMeshDataFixer:")
    if fixer_position != -1:
        # Add __init__.py creation to the constructor or init method
        init_patch = '''
    
    @staticmethod
    def ensure_package_structure(base_path: Path):
        """Ensure proper Python package structure"""
        # Ensure __init__.py exists in ingest_pdf
        ingest_pdf_dir = base_path / "ingest_pdf"
        if ingest_pdf_dir.exists():
            init_file = ingest_pdf_dir / "__init__.py"
            if not init_file.exists():
                init_file.write_text("")
                print("✅ Created __init__.py in ingest_pdf")
'''
        # Find a good place to insert this
        fix_method_pos = content.find("def fix_concept_mesh_data(data_path: Path) -> bool:")
        if fix_method_pos != -1:
            # Add a call to ensure_package_structure at the beginning of __init__
            init_pos = content.find("def __init__(self", fixer_position)
            if init_pos != -1 and init_pos < fix_method_pos:
                # Find the end of __init__ method
                next_line = content.find("\n", init_pos)
                init_body_start = content.find("\n", next_line) + 1
                
                # Add the call
                package_fix = "        # Ensure package structure for imports\n        ConceptMeshDataFixer.ensure_package_structure(self.script_dir)\n"
                
                new_content = new_content[:init_body_start] + package_fix + new_content[init_body_start:]
    
    # Save the patched launcher
    backup_path = launcher_path.with_suffix('.py.backup')
    launcher_path.rename(backup_path)
    print(f"📁 Created backup: {backup_path}")
    
    with open(launcher_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Successfully patched enhanced_launcher.py!")
    print("🚀 You can now run: python enhanced_launcher.py")
    
    return True

def create_init_files():
    """Create necessary __init__.py files"""
    script_dir = Path(__file__).parent.absolute()
    
    # Create __init__.py in ingest_pdf if needed
    ingest_pdf_dir = script_dir / "ingest_pdf"
    if ingest_pdf_dir.exists():
        init_file = ingest_pdf_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")
            print("✅ Created __init__.py in ingest_pdf")
    
    # Create concept_mesh mock if needed
    concept_mesh_dir = script_dir / "concept_mesh"
    if not concept_mesh_dir.exists():
        concept_mesh_dir.mkdir(exist_ok=True)
        
    mock_init = concept_mesh_dir / "__init__.py"
    if not mock_init.exists():
        mock_content = '''# Mock ConceptMeshConnector for testing
class ConceptMeshConnector:
    """Mock ConceptMeshConnector for testing"""
    
    def __init__(self, url=None):
        self.url = url or "http://localhost:8003/api/mesh"
        
    def connect(self):
        """Mock connect method"""
        return True
        
    def get_concepts(self):
        """Mock get_concepts method"""
        return []
        
    def add_concept(self, concept):
        """Mock add_concept method"""
        return {"id": "mock_id", "concept": concept}

__all__ = ['ConceptMeshConnector']
'''
        mock_init.write_text(mock_content)
        print("✅ Created mock concept_mesh module")

if __name__ == "__main__":
    print("🔧 Fixing import issues for enhanced_launcher.py...")
    print("="*50)
    
    # First create necessary files
    create_init_files()
    
    # Then patch the launcher
    if patch_enhanced_launcher():
        print("\n✅ All fixes applied successfully!")
        print("\n📋 What was fixed:")
        print("   1. Added PYTHONPATH configuration for cognitive_interface")
        print("   2. Created __init__.py in ingest_pdf for proper imports")
        print("   3. Created mock concept_mesh module")
        print("   4. Patched enhanced_launcher.py to handle imports properly")
        print("\n🚀 Next step: python enhanced_launcher.py")
    else:
        print("\n❌ Failed to apply fixes!")
