#!/usr/bin/env python3
"""
Quick script to run soft duplicate detection and show results
"""

import subprocess
import json
from pathlib import Path

# Run the soft deduplication
print("🧬 Running soft duplicate detection...")
subprocess.run(["poetry", "run", "python", "soft_dedupe.py"])

# Load and display results
results_file = Path("soft_duplicates.json")
if results_file.exists():
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*60)
    print("📊 SOFT DUPLICATE ANALYSIS SUMMARY")
    print("="*60)
    
    stats = results.get('stats', {})
    print(f"Total concepts analyzed: {stats.get('total_concepts', 0)}")
    print(f"Duplicate pairs found: {stats.get('duplicate_pairs', 0)}")
    print(f"Duplicate clusters: {stats.get('clusters_found', 0)}")
    
    # Show clusters
    clusters = results.get('clusters', [])
    if clusters:
        print("\n🎯 Top Duplicate Clusters:")
        for i, cluster in enumerate(clusters[:5]):
            print(f"\nCluster {i+1} ({cluster['size']} concepts):")
            print(f"  Recommended primary: '{cluster['primary']['label']}'")
            print("  Members:")
            for member in cluster['members']:
                print(f"    - {member['label']} (score: {member.get('score', 0):.3f})")
    
    # Show pairs with high similarity
    pairs = results.get('pairs', [])
    high_similarity = [p for p in pairs if p['similarity'] > 0.95]
    if high_similarity:
        print("\n⚡ Very High Similarity Pairs (>0.95):")
        for pair in high_similarity[:5]:
            print(f"  '{pair['concept1']['label']}' ≈ '{pair['concept2']['label']}' ({pair['similarity']:.3f})")
