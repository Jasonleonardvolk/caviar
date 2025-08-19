#!/usr/bin/env python3
"""
Memory Vault Dashboard
Shows comprehensive statistics and insights about your vault
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def analyze_vault():
    """Comprehensive vault analysis"""
    vault_path = Path("data/memory_vault/memories")
    if not vault_path.exists():
        logger.error("❌ Vault not found")
        return
    
    # Load all concepts
    concepts = []
    memory_files = list(vault_path.glob("*.json"))
    
    for file in memory_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            content = data.get('content', {})
            if isinstance(content, dict) and 'label' in content:
                concepts.append(content)
        except:
            pass
    
    # Analyze
    print("\n" + "="*70)
    print("MEMORY VAULT DASHBOARD")
    print("="*70)
    
    # Basic stats
    print(f"\n📊 Basic Statistics:")
    print(f"  Total concepts: {len(concepts)}")
    print(f"  Average score: {sum(c.get('score', 0) for c in concepts) / len(concepts):.3f}")
    
    # Type distribution
    type_counts = Counter(c.get('metadata', {}).get('concept_type', 'untyped') for c in concepts)
    print(f"\n🏷️ Concept Types:")
    for ctype, count in type_counts.most_common():
        percentage = (count / len(concepts)) * 100
        print(f"  {ctype:<12} {count:>3} ({percentage:>5.1f}%) {'█' * int(percentage/2)}")
    
    # Method distribution
    method_counts = Counter()
    for c in concepts:
        methods = c.get('method', 'unknown').split('+')
        for m in methods:
            method_counts[m.strip()] += 1
    
    print(f"\n🛠️ Extraction Methods:")
    for method, count in method_counts.most_common():
        print(f"  {method:<15} {count:>3}")
    
    # Relationship analysis
    total_rels = sum(len(c.get('metadata', {}).get('relationships', [])) for c in concepts)
    concepts_with_rels = sum(1 for c in concepts if c.get('metadata', {}).get('relationships'))
    
    print(f"\n🔗 Relationships:")
    print(f"  Total relationships: {total_rels}")
    print(f"  Concepts with relationships: {concepts_with_rels}/{len(concepts)} ({concepts_with_rels/len(concepts)*100:.1f}%)")
    
    # Relationship types
    rel_types = Counter()
    for c in concepts:
        for rel in c.get('metadata', {}).get('relationships', []):
            rel_types[rel.get('type', 'unknown')] += 1
    
    if rel_types:
        print(f"  Relationship types:")
        for rtype, count in rel_types.most_common():
            print(f"    {rtype:<15} {count:>3}")
    
    # Score distribution
    print(f"\n📈 Score Distribution:")
    score_ranges = {
        '0.9-1.0': 0,
        '0.8-0.9': 0,
        '0.7-0.8': 0,
        '<0.7': 0
    }
    
    for c in concepts:
        score = c.get('score', 0)
        if score >= 0.9:
            score_ranges['0.9-1.0'] += 1
        elif score >= 0.8:
            score_ranges['0.8-0.9'] += 1
        elif score >= 0.7:
            score_ranges['0.7-0.8'] += 1
        else:
            score_ranges['<0.7'] += 1
    
    for range_name, count in score_ranges.items():
        percentage = (count / len(concepts)) * 100
        print(f"  {range_name:<8} {count:>3} ({percentage:>5.1f}%) {'█' * int(percentage/2)}")
    
    # Top concepts by score
    top_concepts = sorted(concepts, key=lambda x: x.get('score', 0), reverse=True)[:10]
    print(f"\n🏆 Top 10 Concepts by Score:")
    for i, c in enumerate(top_concepts):
        print(f"  {i+1:>2}. {c['label']:<30} {c.get('score', 0):.3f} [{c.get('metadata', {}).get('concept_type', 'untyped')}]")
    
    # Cluster analysis
    word_freq = Counter()
    for c in concepts:
        words = c['label'].lower().split()
        for word in words:
            if len(word) > 3:
                word_freq[word] += 1
    
    print(f"\n🌐 Most Common Terms:")
    for word, count in word_freq.most_common(10):
        print(f"  '{word}': {count} occurrences")
    
    # Normalization status
    normalized_count = sum(1 for c in concepts if 'normalized_label' in c.get('metadata', {}))
    if normalized_count > 0:
        print(f"\n🔄 Synonym Normalization:")
        print(f"  Concepts normalized: {normalized_count}/{len(concepts)} ({normalized_count/len(concepts)*100:.1f}%)")
    
    # Health indicators
    print(f"\n💚 Vault Health Indicators:")
    health_score = 100
    issues = []
    
    if len(concepts) < 20:
        issues.append("Low concept count")
        health_score -= 20
    
    if concepts_with_rels / len(concepts) < 0.3:
        issues.append("Low relationship coverage")
        health_score -= 15
    
    if type_counts.get('untyped', 0) > len(concepts) * 0.5:
        issues.append("Many untyped concepts")
        health_score -= 10
    
    print(f"  Health Score: {health_score}/100")
    if issues:
        print("  Issues:")
        for issue in issues:
            print(f"    ⚠️ {issue}")
    else:
        print("  ✅ All indicators healthy!")
    
    # Recent additions (if timestamps available)
    concepts_with_time = [c for c in concepts if 'created_at' in c.get('metadata', {})]
    if concepts_with_time:
        recent = sorted(concepts_with_time, 
                       key=lambda x: x['metadata']['created_at'], 
                       reverse=True)[:5]
        print(f"\n🆕 Recently Added:")
        for c in recent:
            print(f"  - {c['label']} ({c['metadata']['created_at'][:10]})")


if __name__ == "__main__":
    analyze_vault()
    
    # Check for additional reports
    if Path("soft_duplicates.json").exists():
        print("\n💡 Additional reports available:")
        print("  - soft_duplicates.json (run: python soft_dedupe.py)")
    
    if Path("concept_graph.json").exists():
        print("  - concept_graph.json (visualization data)")
    
    audit_dir = Path("data/memory_vault/audits")
    if audit_dir.exists() and list(audit_dir.glob("*.json")):
        print("  - Audit reports in data/memory_vault/audits/")
