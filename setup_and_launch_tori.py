#!/usr/bin/env python3
"""
Pre-launcher Setup for TORI
===========================

This script sets up the environment properly before launching TORI.
It fixes the cognitive_interface and concept_mesh import issues.

Usage: python setup_and_launch_tori.py
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Setup the environment for proper imports"""
    
    # Get the project root
    project_root = Path(__file__).parent.absolute()
    
    print("🔧 Setting up TORI environment...")
    print(f"📂 Project root: {project_root}")
    
    # 1. Add project root to PYTHONPATH
    os.environ['PYTHONPATH'] = str(project_root)
    sys.path.insert(0, str(project_root))
    
    # 2. Create __init__.py in ingest_pdf if missing
    ingest_pdf_dir = project_root / "ingest_pdf"
    if ingest_pdf_dir.exists():
        init_file = ingest_pdf_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")
            print("✅ Created __init__.py in ingest_pdf")
    
    # 3. Create mock concept_mesh if needed
    concept_mesh_dir = project_root / "concept_mesh"
    if not concept_mesh_dir.exists():
        concept_mesh_dir.mkdir(exist_ok=True)
        print("📁 Created concept_mesh directory")
    
    mock_init = concept_mesh_dir / "__init__.py"
    if not mock_init.exists() or mock_init.stat().st_size == 0:
        mock_content = '''# Mock ConceptMeshConnector module
"""
Mock implementation of ConceptMeshConnector for testing.
This will be replaced when the real concept-mesh service is available.
"""

class ConceptMeshConnector:
    """Mock ConceptMeshConnector for testing"""
    
    def __init__(self, url=None):
        self.url = url or "http://localhost:8003/api/mesh"
        self.connected = False
        self.concepts = []
        
    def connect(self):
        """Mock connect method"""
        self.connected = True
        return True
        
    def disconnect(self):
        """Mock disconnect method"""
        self.connected = False
        return True
        
    def get_concepts(self):
        """Mock get_concepts method"""
        return self.concepts
        
    def add_concept(self, concept):
        """Mock add_concept method"""
        concept_id = f"mock_{len(self.concepts) + 1}"
        concept_data = {
            "id": concept_id,
            "concept": concept,
            "timestamp": "2025-01-28T00:00:00Z"
        }
        self.concepts.append(concept_data)
        return concept_data
        
    def update_concept(self, concept_id, updates):
        """Mock update_concept method"""
        for concept in self.concepts:
            if concept["id"] == concept_id:
                concept.update(updates)
                return concept
        return None
        
    def delete_concept(self, concept_id):
        """Mock delete_concept method"""
        self.concepts = [c for c in self.concepts if c["id"] != concept_id]
        return True

# Make it importable
__all__ = ['ConceptMeshConnector']
'''
        mock_init.write_text(mock_content)
        print("✅ Created mock concept_mesh module")
    
    # 4. Test imports
    print("\n🧪 Testing imports...")
    
    # Test cognitive_interface
    try:
        from ingest_pdf.cognitive_interface import add_concept_diff
        print("✅ cognitive_interface import successful")
    except ImportError as e:
        print(f"⚠️  cognitive_interface import failed: {e}")
        print("   This is okay if the service will be started separately")
    
    # Test concept_mesh
    try:
        from python.core import ConceptMeshConnector
        print("✅ concept_mesh import successful")
    except ImportError as e:
        print(f"❌ concept_mesh import failed: {e}")
    
    print("\n✅ Environment setup complete!")

def launch_tori():
    """Launch TORI with the enhanced launcher"""
    
    project_root = Path(__file__).parent.absolute()
    launcher_path = project_root / "enhanced_launcher.py"
    
    if not launcher_path.exists():
        print("❌ enhanced_launcher.py not found!")
        return False
    
    print("\n🚀 Launching TORI...")
    print("="*60)
    
    # Launch with the same Python interpreter and environment
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    try:
        # Run the enhanced launcher
        subprocess.run(
            [sys.executable, str(launcher_path)],
            env=env,
            cwd=str(project_root)
        )
    except KeyboardInterrupt:
        print("\n👋 TORI shutdown requested")
    except Exception as e:
        print(f"\n❌ Error launching TORI: {e}")
        return False
    
    return True

def main():
    """Main entry point"""
    print("🌊 TORI Pre-launcher Setup")
    print("="*60)
    
    # Setup environment
    setup_environment()
    
    # Launch TORI
    launch_tori()

if __name__ == "__main__":
    main()
