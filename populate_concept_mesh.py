#!/usr/bin/env python3
"""
Populate concept mesh with seed concepts at startup
This ensures the lattice has something to work with
"""

import os
import json
import shutil
from datetime import datetime

def find_concept_mesh_initialization():
    """Find where concept mesh is initialized"""
    
    # Common locations for concept mesh initialization
    potential_files = [
        "python/core/concept_mesh.py",
        "python/core/ConceptMesh.py",
        "concept_mesh/src/lib.rs",
        "ingest_pdf/pipeline/quality.py",
        "enhanced_launcher.py"
    ]
    
    for file_path in potential_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if "concept_mesh" in content.lower() or "ConceptMesh" in content:
                    print(f"Found concept mesh code in: {file_path}")
                    return file_path, content
    
    return None, None

def add_seed_loading_to_concept_mesh():
    """Add seed loading to ConceptMesh class"""
    
    mesh_file = "python/core/ConceptMesh.py"
    
    if not os.path.exists(mesh_file):
        # Try lowercase
        mesh_file = "python/core/concept_mesh.py"
    
    if not os.path.exists(mesh_file):
        print(f"Could not find {mesh_file}")
        return False
    
    print(f"Adding seed loading to {mesh_file}")
    
    with open(mesh_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already has seed loading
    if "load_seeds" in content:
        print("Already has load_seeds method")
        return True
    
    # Find the class definition
    class_pos = content.find("class ConceptMesh")
    if class_pos == -1:
        print("Could not find ConceptMesh class")
        return False
    
    # Find __init__ method
    init_pos = content.find("def __init__", class_pos)
    if init_pos == -1:
        print("Could not find __init__ method")
        return False
    
    # Find the end of __init__ to add seed loading
    init_end = content.find("\n\n    def", init_pos)
    if init_end == -1:
        init_end = content.find("\n\nclass", init_pos)
    if init_end == -1:
        init_end = len(content)
    
    # Add load_seeds method
    seed_loading_code = '''
    def load_seeds(self, seed_file: str = None):
        """Load seed concepts from file"""
        if seed_file is None:
            # Default seed files
            seed_files = [
                "data/concept_seed_universal.json",
                "ingest_pdf/data/concept_seed_universal.json",
                "data/seed_concepts.json"
            ]
            
            for sf in seed_files:
                if os.path.exists(sf):
                    seed_file = sf
                    break
        
        if not seed_file or not os.path.exists(seed_file):
            logger.warning("No seed file found")
            return 0
        
        try:
            with open(seed_file, 'r', encoding='utf-8') as f:
                seeds = json.load(f)
            
            count = 0
            for seed in seeds:
                if isinstance(seed, dict) and 'name' in seed:
                    concept_id = seed.get('id', seed['name'].lower().replace(' ', '_'))
                    self.add_concept(concept_id, seed['name'], seed)
                    count += 1
            
            logger.info(f"Loaded {count} seed concepts from {seed_file}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to load seeds: {e}")
            return 0
    
    def ensure_populated(self):
        """Ensure mesh has at least some concepts"""
        if self.count() == 0:
            loaded = self.load_seeds()
            if loaded > 0:
                logger.info("Concept mesh populated with %d seed concepts", loaded)
                # Trigger events for lattice
                for concept_id, concept in self.concepts.items():
                    if hasattr(self, 'event_bus') and self.event_bus:
                        self.event_bus.publish("concept_added", {
                            "id": concept_id,
                            "name": concept.get("name", concept_id),
                            "data": concept
                        })
            else:
                logger.warning("Concept mesh remains empty - no seeds loaded")
'''
    
    # Insert the code
    new_content = content[:init_end] + seed_loading_code + content[init_end:]
    
    # Also add imports if needed
    if "import json" not in content:
        import_pos = content.find("import ")
        if import_pos != -1:
            next_line = content.find("\n", import_pos)
            new_content = content[:next_line] + "\nimport json\nimport os" + content[next_line:]
    
    # Create backup
    backup = f"{mesh_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(mesh_file, backup)
    print(f"Created backup: {backup}")
    
    # Write the file
    with open(mesh_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("Added seed loading methods to ConceptMesh")
    return True

def add_startup_population():
    """Add concept mesh population to enhanced_launcher.py"""
    
    launcher_file = "enhanced_launcher.py"
    
    if not os.path.exists(launcher_file):
        print(f"Could not find {launcher_file}")
        return False
    
    print(f"Adding startup population to {launcher_file}")
    
    with open(launcher_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already has population
    if "ensure_populated" in content or "load_seeds" in content:
        print("Already has concept population")
        return True
    
    # Find where concept mesh is initialized
    mesh_init = content.find("ConceptMesh()")
    if mesh_init == -1:
        print("Could not find ConceptMesh initialization")
        return False
    
    # Find the line after initialization
    next_line = content.find("\n", mesh_init)
    
    # Get proper indentation
    line_start = content.rfind("\n", 0, mesh_init) + 1
    indent = content[line_start:mesh_init].replace("concept_mesh = ", "")
    
    # Add population code
    population_code = f'''
{indent}# Ensure concept mesh has seed concepts
{indent}if hasattr(concept_mesh, 'ensure_populated'):
{indent}    concept_mesh.ensure_populated()
{indent}else:
{indent}    logger.warning("ConceptMesh missing ensure_populated method")
{indent}    # Try manual loading
{indent}    if hasattr(concept_mesh, 'count') and concept_mesh.count() == 0:
{indent}        logger.info("Concept mesh is empty, needs seed concepts")
'''
    
    # Insert the code
    new_content = content[:next_line] + population_code + content[next_line:]
    
    # Create backup
    backup = f"{launcher_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(launcher_file, backup)
    print(f"Created backup: {backup}")
    
    # Write the file
    with open(launcher_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("Added startup population to launcher")
    return True

def create_minimal_seed_file():
    """Create a minimal seed concepts file if none exists"""
    
    seed_data = [
        {"name": "artificial_intelligence", "priority": 0.9, "category": "technology"},
        {"name": "machine_learning", "priority": 0.85, "category": "technology"},
        {"name": "neural_network", "priority": 0.8, "category": "technology"},
        {"name": "consciousness", "priority": 0.9, "category": "philosophy"},
        {"name": "cognition", "priority": 0.85, "category": "philosophy"},
        {"name": "memory", "priority": 0.8, "category": "neuroscience"},
        {"name": "quantum", "priority": 0.75, "category": "physics"},
        {"name": "emergence", "priority": 0.8, "category": "systems"},
        {"name": "complexity", "priority": 0.75, "category": "systems"},
        {"name": "topology", "priority": 0.7, "category": "mathematics"}
    ]
    
    # Try to create in data directory
    os.makedirs("data", exist_ok=True)
    
    seed_file = "data/seed_concepts.json"
    with open(seed_file, 'w', encoding='utf-8') as f:
        json.dump(seed_data, f, indent=2)
    
    print(f"Created minimal seed file: {seed_file}")
    return seed_file

def main():
    print("Fixing Empty Concept Mesh at Startup")
    print("=" * 60)
    print("\nProblem: Concept mesh has 0 concepts, lattice has nothing to oscillate")
    print("Solution: Load seed concepts at startup\n")
    
    # First ensure we have seed concepts
    if not any(os.path.exists(f) for f in ["data/seed_concepts.json", 
                                            "ingest_pdf/data/concept_seed_universal.json",
                                            "data/concept_seed_universal.json"]):
        print("No seed file found, creating minimal one...")
        create_minimal_seed_file()
    
    # Add seed loading to ConceptMesh
    if add_seed_loading_to_concept_mesh():
        print("\nStep 1: Added seed loading to ConceptMesh class")
    else:
        print("\nStep 1: Could not modify ConceptMesh class")
    
    # Add startup population
    if add_startup_population():
        print("Step 2: Added startup population to launcher")
    else:
        print("Step 2: Could not modify launcher")
    
    print("\nWhat this does:")
    print("1. Adds load_seeds() method to ConceptMesh")
    print("2. Adds ensure_populated() to load seeds if mesh is empty")
    print("3. Triggers concept_added events for the lattice")
    print("4. Creates a minimal seed file if none exists")
    
    print("\nAfter restart:")
    print("- Concept mesh will have at least 10 seed concepts")
    print("- Lattice will have something to oscillate")
    print("- Events will fire to populate the system")
    
    print("\nNOTE: No visual needed on homepage - this is just for backend operation!")

if __name__ == "__main__":
    main()
