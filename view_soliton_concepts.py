# view_soliton_concepts.py - View what concepts were extracted
import json
from pathlib import Path

archive_id = "archive_1752186498381_958672"
archive_path = Path(f"./psi_archive/{archive_id}.json")

if archive_path.exists():
    with open(archive_path, 'r') as f:
        data = json.load(f)
    
    print("🎯 Living Soliton Memory Systems - Extracted Concepts")
    print("=" * 60)
    print(f"Source: {data['source_path']}")
    print(f"Concepts extracted: {data['concepts_count']}")
    print(f"Timestamp: {data['timestamp']}")
    
    # The actual concepts might not be in the archive (just the count)
    # But we can see the mesh delta
    if 'mesh_delta' in data:
        mesh = data['mesh_delta']
        print(f"\n📊 Concept Mesh Update:")
        print(f"  - Nodes added: {mesh.get('nodes_added', 0)}")
        print(f"  - Edges added: {mesh.get('edges_added', 0)}")
        print(f"  - Similarity threshold: {mesh.get('similarity_threshold', 0)}")
    
    if 'penrose_stats' in data:
        penrose = data['penrose_stats']
        print(f"\n🔍 Penrose Verification Details:")
        if 'vector_quality' in penrose:
            for check, passed in penrose['vector_quality'].items():
                status = "✅" if passed else "❌"
                print(f"  - {check}: {status}")
else:
    print(f"Archive file not found: {archive_path}")
