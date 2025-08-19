#!/usr/bin/env python3
"""
Concept Data Consolidation Script
=================================
Consolidates all concept data into a single canonical source: data/concept_db.json
Archives old files and separates graph edges from concept entities.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
import hashlib

# Define file paths
DATA_DIR = Path("data")
CONCEPT_DB_PATH = DATA_DIR / "concept_db.json"
CONCEPT_GRAPH_PATH = Path("concept_graph.json")
SOLITON_MEMORY_PATH = Path("soliton_concept_memory.json")
ARCHIVE_DIR = DATA_DIR / "archive"

# Create directories if needed
DATA_DIR.mkdir(exist_ok=True)
ARCHIVE_DIR.mkdir(exist_ok=True)

def generate_concept_id(name):
    """Generate a deterministic ID for a concept based on its name"""
    return f"concept_{hashlib.md5(name.lower().encode()).hexdigest()[:12]}"

def load_json_safe(filepath):
    """Safely load JSON file, return empty dict if not found"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return {}
    except json.JSONDecodeError as e:
        print(f"JSON decode error in {filepath}: {e}")
        return {}

def migrate_concepts():
    """Main migration function"""
    print("🚀 Starting Concept Data Consolidation...")
    print("="*60)
    
    # Step 1: Load all existing data
    print("\n📂 Loading existing files...")
    
    # Load concept_db.json (simple array format)
    old_concept_db = []
    if CONCEPT_DB_PATH.exists():
        old_concept_db = load_json_safe(CONCEPT_DB_PATH)
        print(f"  ✓ Loaded {len(old_concept_db)} concepts from concept_db.json")
    
    # Load concept_graph.json (has nodes and edges)
    concept_graph = {}
    if CONCEPT_GRAPH_PATH.exists():
        concept_graph = load_json_safe(CONCEPT_GRAPH_PATH)
        print(f"  ✓ Loaded {len(concept_graph.get('nodes', []))} nodes from concept_graph.json")
        print(f"  ✓ Found {len(concept_graph.get('edges', []))} edges to preserve")
    
    # Load soliton_concept_memory.json (appears to be empty)
    soliton_memory = {}
    if SOLITON_MEMORY_PATH.exists():
        soliton_memory = load_json_safe(SOLITON_MEMORY_PATH)
        print(f"  ✓ Loaded soliton_concept_memory.json")
    
    # Step 2: Create canonical concept structure
    print("\n🔨 Building canonical concept database...")
    canonical_concepts = {}
    
    # Process concepts from old concept_db.json
    for concept in old_concept_db:
        if isinstance(concept, dict) and 'name' in concept:
            concept_id = generate_concept_id(concept['name'])
            canonical_concepts[concept_id] = {
                "id": concept_id,
                "name": concept['name'],
                "priority": concept.get('priority', 0.5),
                "category": concept.get('category', 'general'),
                "boost_multiplier": concept.get('boost_multiplier', 1.0),
                "score": concept.get('priority', 0.5),  # Use priority as score
                "method": "manual",  # These were manually added
                "created_at": datetime.utcnow().isoformat(),
                "metadata": {
                    "source": "original_concept_db"
                }
            }
    
    print(f"  ✓ Processed {len(canonical_concepts)} concepts from concept_db.json")
    
    # Process concepts from concept_graph.json nodes
    if 'nodes' in concept_graph:
        new_concepts = 0
        for node in concept_graph['nodes']:
            concept_id = node.get('id', generate_concept_id(node.get('label', 'unknown')))
            
            # Check if concept already exists (by name)
            existing = False
            for cid, c in canonical_concepts.items():
                if c['name'].lower() == node.get('label', '').lower():
                    existing = True
                    # Update with additional metadata
                    c['score'] = max(c.get('score', 0), node.get('score', 0))
                    c['method'] = node.get('method', c.get('method', 'unknown'))
                    c['metadata']['graph_id'] = node.get('id')
                    break
            
            if not existing:
                canonical_concepts[concept_id] = {
                    "id": concept_id,
                    "name": node.get('label', 'Unknown'),
                    "score": node.get('score', 0.5),
                    "method": node.get('method', 'unknown'),
                    "category": "extracted",  # These were extracted
                    "priority": node.get('score', 0.5),
                    "boost_multiplier": 1.0,
                    "created_at": datetime.utcnow().isoformat(),
                    "metadata": {
                        "source": "concept_graph",
                        "relationships_count": node.get('relationships_count', 0)
                    }
                }
                new_concepts += 1
        
        print(f"  ✓ Added {new_concepts} new concepts from concept_graph.json")
    
    # Step 3: Create the new canonical structure
    canonical_db = {
        "version": "2.0",
        "schema": "canonical_concept_db",
        "created_at": datetime.utcnow().isoformat(),
        "metadata": {
            "description": "Canonical concept database for TORI system",
            "migration_date": datetime.utcnow().isoformat(),
            "total_concepts": len(canonical_concepts),
            "sources_merged": ["concept_db.json", "concept_graph.json"]
        },
        "concepts": canonical_concepts
    }
    
    # Step 4: Archive old files
    print("\n📦 Archiving old files...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Archive old concept_db.json
    if CONCEPT_DB_PATH.exists():
        archive_path = ARCHIVE_DIR / f"concept_db_backup_{timestamp}.json"
        shutil.copy2(CONCEPT_DB_PATH, archive_path)
        print(f"  ✓ Archived concept_db.json to {archive_path}")
    
    # Archive old concept_graph.json (full version)
    if CONCEPT_GRAPH_PATH.exists():
        archive_path = ARCHIVE_DIR / f"concept_graph_full_{timestamp}.json"
        shutil.copy2(CONCEPT_GRAPH_PATH, archive_path)
        print(f"  ✓ Archived concept_graph.json to {archive_path}")
    
    # Archive soliton_concept_memory.json
    if SOLITON_MEMORY_PATH.exists():
        archive_path = ARCHIVE_DIR / f"soliton_concept_memory_{timestamp}.json"
        shutil.move(str(SOLITON_MEMORY_PATH), str(archive_path))
        print(f"  ✓ Archived and removed soliton_concept_memory.json")
    
    # Step 5: Write new canonical concept_db.json
    print("\n💾 Writing new canonical files...")
    with open(CONCEPT_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(canonical_db, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Wrote {len(canonical_concepts)} concepts to {CONCEPT_DB_PATH}")
    
    # Step 6: Update concept_graph.json to contain ONLY edges
    if 'edges' in concept_graph:
        graph_only = {
            "version": "2.0",
            "schema": "concept_graph_edges_only",
            "created_at": datetime.utcnow().isoformat(),
            "metadata": {
                "description": "Graph edges/relationships between concepts",
                "note": "Concept entities are stored in data/concept_db.json"
            },
            "edges": concept_graph['edges'],
            "stats": {
                "total_edges": len(concept_graph['edges'])
            }
        }
        
        with open(CONCEPT_GRAPH_PATH, 'w', encoding='utf-8') as f:
            json.dump(graph_only, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Updated concept_graph.json with {len(concept_graph['edges'])} edges only")
    
    # Step 7: Create summary report
    print("\n📊 Migration Summary:")
    print(f"  • Total concepts consolidated: {len(canonical_concepts)}")
    print(f"  • Original concept_db.json concepts: {len(old_concept_db)}")
    print(f"  • Concepts from graph: {len(concept_graph.get('nodes', []))}")
    print(f"  • Graph edges preserved: {len(concept_graph.get('edges', []))}")
    print(f"  • Files archived: 3")
    print(f"\n✅ Migration complete! Your canonical concept source is: {CONCEPT_DB_PATH}")
    
    # Create a migration report
    report = {
        "migration_timestamp": datetime.utcnow().isoformat(),
        "concepts_migrated": len(canonical_concepts),
        "files_processed": {
            "concept_db.json": len(old_concept_db),
            "concept_graph.json": len(concept_graph.get('nodes', [])),
            "soliton_concept_memory.json": "empty (archived)"
        },
        "edges_preserved": len(concept_graph.get('edges', [])),
        "canonical_file": str(CONCEPT_DB_PATH),
        "archive_location": str(ARCHIVE_DIR)
    }
    
    report_path = DATA_DIR / f"migration_report_{timestamp}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Migration report saved to: {report_path}")

if __name__ == "__main__":
    print("🎯 TORI Concept Data Consolidation Tool")
    print("="*60)
    print("This will consolidate all concept data into:")
    print(f"  → {CONCEPT_DB_PATH}")
    print("\nOld files will be archived to:")
    print(f"  → {ARCHIVE_DIR}")
    print("="*60)
    
    response = input("\nProceed with migration? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        migrate_concepts()
    else:
        print("Migration cancelled.")
