"""
Concept mesh loader utility
"""
import json
from pathlib import Path

def load_mesh():
    """Load the current concept mesh from data.json"""
    concepts_path = Path(__file__).parent / "data.json"
    
    if concepts_path.exists():
        with open(concepts_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Return empty mesh if file doesn't exist
    return {
        "concepts": {},
        "metadata": {
            "version": "1.0",
            "last_updated": None
        }
    }

def save_mesh(mesh_data):
    """Save the concept mesh to data.json"""
    concepts_path = Path(__file__).parent / "data.json"
    
    with open(concepts_path, 'w', encoding='utf-8') as f:
        json.dump(mesh_data, f, indent=2)
