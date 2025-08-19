#!/usr/bin/env python3
"""
Fix lattice oscillators - ensure concepts are loaded and converted to oscillators
"""
import sys
import os
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_and_fix_lattice():
    """Check why lattice has 0 oscillators and fix it"""
    
    print("\n[DIAGNOSE] Checking lattice oscillator issue...\n")
    
    # Step 1: Check concept mesh data
    print("[1] Checking concept mesh data...")
    concept_mesh_file = Path("concept_mesh/data.json")
    
    if concept_mesh_file.exists():
        with open(concept_mesh_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'concepts' in data:
            concepts = data['concepts']
            print(f"   [OK] Found {len(concepts)} concepts in mesh file")
            
            # Check concept structure
            for i, concept in enumerate(concepts[:3]):
                print(f"   - Concept {i+1}: {concept.get('name', 'unnamed')}")
                print(f"     id: {concept.get('id', 'no-id')}")
                print(f"     strength: {concept.get('strength', 0)}")
                print(f"     embedding: {len(concept.get('embedding', []))} dims")
        else:
            print("   [ERROR] Invalid concept mesh structure")
            return False
    else:
        print("   [ERROR] Concept mesh file not found")
        return False
    
    # Step 2: Try to load the lattice directly
    print("\n[2] Attempting to access lattice directly...")
    try:
        # Try importing the components
        from python.core.cognitive_interface import CognitiveInterface
        from python.core.concept_mesh import ConceptMesh
        
        print("   [OK] Imported cognitive components")
        
        # Get the cognitive interface
        ci = CognitiveInterface()
        print("   [OK] Created CognitiveInterface")
        
        # Check the concept mesh
        mesh = ci.concept_mesh
        print(f"   [INFO] Concept mesh has {len(mesh.concepts) if hasattr(mesh, 'concepts') else 0} concepts in memory")
        
        # Try to manually load concepts
        if hasattr(mesh, 'load_from_file') or hasattr(mesh, 'load'):
            print("   [INFO] Attempting to load concepts from file...")
            try:
                if hasattr(mesh, 'load_from_file'):
                    mesh.load_from_file(concept_mesh_file)
                elif hasattr(mesh, 'load'):
                    mesh.load(concept_mesh_file)
                print(f"   [OK] Loaded concepts, mesh now has {len(mesh.concepts) if hasattr(mesh, 'concepts') else 0} concepts")
            except Exception as e:
                print(f"   [ERROR] Failed to load concepts: {e}")
        
        # Check if lattice exists
        if hasattr(ci, 'lattice'):
            lattice = ci.lattice
            print(f"   [INFO] Lattice has {getattr(lattice, 'oscillator_count', 0)} oscillators")
            
            # Try to rebuild
            if hasattr(lattice, 'rebuild'):
                print("   [INFO] Attempting lattice rebuild...")
                try:
                    result = lattice.rebuild(full=True)
                    print(f"   [OK] Rebuild result: {result}")
                except Exception as e:
                    print(f"   [ERROR] Rebuild failed: {e}")
        else:
            print("   [WARNING] No lattice found in CognitiveInterface")
            
    except ImportError as e:
        print(f"   [ERROR] Import failed: {e}")
        print("\n[3] Creating manual lattice fix...")
        create_manual_fix()
    except Exception as e:
        print(f"   [ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    return True

def create_manual_fix():
    """Create a manual fix to ensure concepts are loaded"""
    
    fix_content = '''#!/usr/bin/env python3
"""Manual lattice oscillator seeder"""
import json
from pathlib import Path

# Read concepts
concept_file = Path("concept_mesh/data.json")
with open(concept_file, 'r') as f:
    data = json.load(f)

concepts = data.get('concepts', [])
print(f"[INFO] Found {len(concepts)} concepts to seed")

# Create oscillator initialization file
oscillator_data = {
    "oscillators": [],
    "concept_oscillators": {}
}

for concept in concepts:
    oscillator = {
        "id": concept.get('id'),
        "name": concept.get('name'),
        "frequency": 1.0,
        "amplitude": concept.get('strength', 1.0),
        "phase": 0.0,
        "active": True
    }
    oscillator_data["oscillators"].append(oscillator)
    oscillator_data["concept_oscillators"][concept.get('id')] = oscillator

# Save oscillator data
output_file = Path("data/lattice_oscillators.json")
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(oscillator_data, f, indent=2)

print(f"[OK] Created oscillator data with {len(oscillator_data['oscillators'])} oscillators")
print(f"   Saved to: {output_file}")
'''
    
    with open("seed_lattice_oscillators.py", 'w') as f:
        f.write(fix_content)
    
    print("   [OK] Created seed_lattice_oscillators.py")
    print("   Run: python seed_lattice_oscillators.py")

def main():
    """Main entry point"""
    print("="*60)
    print("LATTICE OSCILLATOR FIXER")
    print("="*60)
    
    success = check_and_fix_lattice()
    
    print("\n" + "="*60)
    print("[SUMMARY]")
    print("="*60)
    
    print("\nThe issue is that concepts exist in the mesh file but aren't")
    print("being converted to oscillators in the lattice.")
    
    print("\n[SOLUTIONS]:")
    print("1. Restart TORI and watch for 'Loading concepts from storage' messages")
    print("2. Run: python seed_lattice_oscillators.py (if created)")
    print("3. Check for filtering rules in the lattice configuration")
    print("4. Ensure concept mesh is loaded before lattice rebuild")
    
    print("\n[NEXT STEPS]:")
    print("1. Stop TORI (Ctrl+C)")
    print("2. Run: python fix_lattice_oscillators.py")
    print("3. Restart TORI: poetry run python enhanced_launcher.py")
    print("4. Watch logs for 'oscillators > 0'")

if __name__ == "__main__":
    main()
