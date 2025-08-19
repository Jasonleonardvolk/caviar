#!/usr/bin/env python3
"""
Test script for VaultInspector mesh comparison and delta tracking
"""

import json
import tempfile
from pathlib import Path
import subprocess
import sys

def create_test_mesh():
    """Create a test ConceptMesh file"""
    mesh_data = {
        "concepts": [
            {"id": "concept_001", "name": "consciousness", "type": "abstract"},
            {"id": "concept_002", "name": "memory", "type": "cognitive"},
            {"id": "concept_003", "name": "learning", "type": "process"},
            {"id": "concept_orphan", "name": "orphaned", "type": "test"}
        ]
    }
    
    mesh_file = Path("test_mesh.json")
    with open(mesh_file, 'w') as f:
        json.dump(mesh_data, f, indent=2)
    
    return mesh_file

def create_test_snapshots():
    """Create test snapshot files"""
    # Old snapshot
    old_snapshot = {
        "memories": {
            "semantic": [
                {
                    "id": "mem_001",
                    "type": "semantic",
                    "content": "The sky is blue",
                    "metadata": {"concept_ids": ["concept_001"]},
                    "importance": 0.8
                },
                {
                    "id": "mem_002",
                    "type": "semantic",
                    "content": "Water is wet",
                    "metadata": {},
                    "importance": 0.5
                }
            ]
        }
    }
    
    # New snapshot - modified, added, and deleted entries
    new_snapshot = {
        "memories": {
            "semantic": [
                {
                    "id": "mem_001",
                    "type": "semantic",
                    "content": "The sky is blue",
                    "metadata": {"concept_ids": ["concept_001", "concept_002"]},
                    "importance": 0.9  # Changed
                },
                {
                    "id": "mem_003",  # New
                    "type": "semantic",
                    "content": "Grass is green",
                    "metadata": {"concept_ids": ["concept_missing"]},  # Missing concept
                    "importance": 0.7
                }
                # mem_002 deleted
            ]
        }
    }
    
    old_file = Path("test_snapshot_old.json")
    new_file = Path("test_snapshot_new.json")
    
    with open(old_file, 'w') as f:
        json.dump(old_snapshot, f, indent=2)
    
    with open(new_file, 'w') as f:
        json.dump(new_snapshot, f, indent=2)
    
    return old_file, new_file

def test_mesh_comparison():
    """Test vault-mesh comparison"""
    print("🧪 Testing Vault-Mesh Comparison...")
    print("=" * 50)
    
    mesh_file = create_test_mesh()
    
    # Run comparison
    result = subprocess.run(
        [sys.executable, "vault_inspector.py", "--compare-mesh", str(mesh_file), "--json"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        comparison = json.loads(result.stdout)
        print(f"✅ Comparison completed")
        print(f"   Orphaned concepts: {comparison['statistics']['orphaned_concepts']}")
        print(f"   Missing concepts: {len(comparison['vault_entries_without_mesh'])}")
    else:
        print(f"❌ Comparison failed: {result.stderr}")
    
    # Cleanup
    mesh_file.unlink()
    
    print()

def test_delta_tracking():
    """Test snapshot delta tracking"""
    print("🧪 Testing Delta Tracking...")
    print("=" * 50)
    
    old_file, new_file = create_test_snapshots()
    
    # Run delta comparison
    result = subprocess.run(
        [sys.executable, "vault_inspector.py", "--delta", str(old_file), str(new_file), "--json"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        delta = json.loads(result.stdout)
        summary = delta['hash_summary']
        print(f"✅ Delta analysis completed")
        print(f"   New entries: {summary['entries_added']}")
        print(f"   Modified entries: {summary['entries_modified']}")
        print(f"   Deleted entries: {summary['entries_deleted']}")
        print(f"   Unchanged entries: {summary['entries_unchanged']}")
    else:
        print(f"❌ Delta analysis failed: {result.stderr}")
    
    # Cleanup
    old_file.unlink()
    new_file.unlink()

if __name__ == "__main__":
    print("🧪 VaultInspector New Features Test")
    print("=" * 50)
    print()
    
    test_mesh_comparison()
    test_delta_tracking()
    
    print("\n✅ All tests completed!")
