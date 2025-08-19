"""
Seed Concept Mesh with Initial Data

This script seeds the concept mesh with initial concept data to ensure
the oscillator lattice has something to work with.
"""

import json
import os
from pathlib import Path
from datetime import datetime

# Ensure entropy pruning is enabled
os.environ['TORI_ENABLE_ENTROPY_PRUNING'] = '1'

# Paths to the concept mesh files
CONCEPT_MESH_DATA_PATH = Path(__file__).parent / "concept_mesh_data.json"
CONCEPTS_JSON_PATH = Path(__file__).parent / "concepts.json"

# Create the data directory structure
DATA_DIR = Path(__file__).parent / "data"
CONCEPT_DIFFS_DIR = DATA_DIR / "concept_diffs"
CONCEPT_DIFFS_DIR.mkdir(parents=True, exist_ok=True)

# Sample concepts to seed the mesh with
SEED_CONCEPTS = [
    {
        "name": "Kagome Lattice",
        "quality_score": 0.95,
        "score": 0.92,
        "metadata": {
            "category": "physics",
            "frequency": 5,
            "section": "title"
        }
    },
    {
        "name": "Soliton Memory",
        "quality_score": 0.93,
        "score": 0.90,
        "metadata": {
            "category": "computer_science",
            "frequency": 7,
            "section": "title"
        }
    },
    {
        "name": "Quantum Information",
        "quality_score": 0.87,
        "score": 0.85,
        "metadata": {
            "category": "physics",
            "frequency": 3,
            "section": "body"
        }
    },
    {
        "name": "Oscillation Dynamics",
        "quality_score": 0.84,
        "score": 0.82,
        "metadata": {
            "category": "physics",
            "frequency": 4,
            "section": "body"
        }
    },
    {
        "name": "Neural Network",
        "quality_score": 0.89,
        "score": 0.86,
        "metadata": {
            "category": "computer_science",
            "frequency": 6,
            "section": "body"
        }
    }
]

def seed_concept_mesh_data():
    """Seed the concept_mesh_data.json file with initial concepts."""
    print(f"Seeding concept_mesh_data.json...")
    
    # Initialize or load existing data
    if CONCEPT_MESH_DATA_PATH.exists():
        with open(CONCEPT_MESH_DATA_PATH, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {"concepts": {}, "diffs": [], "metadata": {"version": "1.0"}}
    else:
        data = {"concepts": {}, "diffs": [], "metadata": {"version": "1.0"}}
    
    # Add seed concepts
    for concept in SEED_CONCEPTS:
        concept_id = concept["name"].lower().replace(" ", "_")
        data["concepts"][concept_id] = concept
    
    # Update metadata
    data["metadata"]["updated_at"] = datetime.now().isoformat()
    data["metadata"]["concept_count"] = len(data["concepts"])
    
    # Save the data
    with open(CONCEPT_MESH_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Added {len(SEED_CONCEPTS)} seed concepts to concept_mesh_data.json")
    return data

def seed_concepts_json():
    """Seed the concepts.json file with the same concepts."""
    print(f"Seeding concepts.json...")
    
    # Initialize or load existing data
    if CONCEPTS_JSON_PATH.exists():
        with open(CONCEPTS_JSON_PATH, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {"concepts": [], "metadata": {"version": "1.0"}}
    else:
        data = {"concepts": [], "metadata": {"version": "1.0"}}
    
    # Add seed concepts
    data["concepts"] = SEED_CONCEPTS
    
    # Update metadata
    data["metadata"]["updated_at"] = datetime.now().isoformat()
    
    # Save the data
    with open(CONCEPTS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Added {len(SEED_CONCEPTS)} seed concepts to concepts.json")
    return data

def create_concept_diff():
    """Create a concept diff file in the data/concept_diffs directory."""
    print(f"Creating concept diff file...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diff_file = CONCEPT_DIFFS_DIR / f"diff_seed_{timestamp}.json"
    
    diff_data = {
        "type": "document",
        "title": "Seed Concepts",
        "concepts": SEED_CONCEPTS,
        "summary": f"{len(SEED_CONCEPTS)} seed concepts.",
        "metadata": {
            "source": "manual_seed",
            "timestamp": timestamp,
            "version": "1.0"
        }
    }
    
    with open(diff_file, 'w', encoding='utf-8') as f:
        json.dump(diff_data, f, indent=2)
    
    print(f"Created concept diff file: {diff_file}")
    return diff_file

def rebuild_lattice():
    """Trigger a rebuild of the oscillator lattice."""
    print("Triggering lattice rebuild...")
    try:
        import requests
        response = requests.post("http://localhost:8002/api/lattice/rebuild")
        if response.status_code == 200:
            print("Lattice rebuild successful!")
            return True
        else:
            print(f"Lattice rebuild failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error triggering lattice rebuild: {e}")
        return False

def check_lattice_status():
    """Check the status of the oscillator lattice."""
    print("Checking lattice status...")
    try:
        import requests
        response = requests.get("http://localhost:8002/api/lattice/snapshot")
        if response.status_code == 200:
            data = response.json()
            summary = data.get("summary", {})
            oscillators = summary.get("oscillators", 0)
            concept_oscillators = summary.get("concept_oscillators", 0)
            print(f"Lattice status: oscillators={oscillators} concept_oscillators={concept_oscillators}")
            return oscillators, concept_oscillators
        else:
            print(f"Failed to get lattice status: {response.status_code}")
            return 0, 0
    except Exception as e:
        print(f"Error checking lattice status: {e}")
        return 0, 0

if __name__ == "__main__":
    print("=== SEEDING CONCEPT MESH WITH INITIAL DATA ===")
    
    # Seed the concept mesh data
    seed_concept_mesh_data()
    
    # Seed the concepts.json file
    seed_concepts_json()
    
    # Create a concept diff file
    create_concept_diff()
    
    # Check lattice status before rebuild
    print("\nLattice status before rebuild:")
    check_lattice_status()
    
    # Rebuild the lattice
    rebuild_lattice()
    
    # Wait a moment for the rebuild to take effect
    import time
    print("\nWaiting for lattice rebuild to take effect...")
    time.sleep(5)
    
    # Check lattice status after rebuild
    print("\nLattice status after rebuild:")
    oscillators, concept_oscillators = check_lattice_status()
    
    if oscillators > 0 and concept_oscillators > 0:
        print("\n✅ SUCCESS: Concept mesh has been seeded and lattice has oscillators!")
    else:
        print("\n⚠️ WARNING: Concept mesh has been seeded but lattice may need manual restart.")
        print("Try restarting the oscillator lattice worker or the full MCP stack.")
