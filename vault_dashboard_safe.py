#!/usr/bin/env python3
"""
Memory Vault Dashboard - Windows Safe Version
Shows comprehensive statistics and insights about your vault
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import logging
import sys

# Force UTF-8 encoding for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def safe_print(text):
    """Print with fallback for Unicode issues"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: replace emojis with ASCII
        text = text.encode('ascii', 'replace').decode('ascii')
        print(text)


def analyze_vault():
    """Comprehensive vault analysis"""
    vault_path = Path("data/memory_vault/memories")
    if not vault_path.exists():
        logger.error("X Vault not found")
        return
    
    # Load all concepts
    concepts = []
    memory_files = list(vault_path.glob("*.json"))
    
    for file in memory_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            content = data.get('content', {})
            if isinstance(content, dict) and 'label' in content:
                concepts.append(content)
        except:
            pass
    
    # Analyze
    safe_print("\n" + "="*70)
    safe_print("MEMORY VAULT DASHBOARD")
    safe_print("="*70)
    
    # Basic stats
    safe_print(f"\n[STATS] Basic Statistics:")
    safe_print(f"  Total concepts: {len(concepts)}")
    safe_print(f"  Average score: {sum(c.get('score', 0) for c in concepts) / len(concepts):.3f}")
    
    # Type distribution
    type_counts = Counter(c.get('metadata', {}).get('concept_type', 'untyped') for c in concepts)
    safe_print(f"\n[TYPES] Concept Types:")
    for ctype, count in type_counts.most_common():
        percentage = (count / len(concepts)) * 100
        bar = '#' * int(percentage/2)
        safe_print(f"  {ctype:<12} {count:>3} ({percentage:>5.1f}%) {bar}")
    
    # Method distribution
    method_counts = Counter()
    for c in concepts:
        methods = c.get('method', 'unknown').split('+')
        for m in methods:
            method_counts[m.strip()] += 1
    
    safe_print(f"\n[METHODS] Extraction Methods:")
    for method, count in method_counts.most_common():
        safe_print(f"  {method:<15} {count:>3}")
    
    # Relationship analysis
    total_rels = sum(len(c.get('metadata', {}).get('relationships', [])) for c in concepts)
    concepts_with_rels = sum(1 for c in concepts if c.get('metadata', {}).get('relationships'))
    
    safe_print(f"\n[RELATIONS] Relationships:")
    safe_print(f"  Total relationships: {total_rels}")
    safe_print(f"  Concepts with relationships: {concepts_with_rels}/{len(concepts)} ({concepts_with_rels/len(concepts)*100:.1f}%)")
    
    # Relationship types
    rel_types = Counter()
    for c in concepts:
        for rel in c.get('metadata', {}).get('relationships', []):
            rel_types[rel.get('type', 'unknown')] += 1
    
    if rel_types:
        safe_print(f"  Relationship types:")
        for rtype, count in rel_types.most_common():
            safe_print(f"    {rtype:<15} {count:>3}")
    
    # Score distribution
    safe_print(f"\n[SCORES] Score Distribution:")
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
        bar = '#' * int(percentage/2)
        safe_print(f"  {range_name:<8} {count:>3} ({percentage:>5.1f}%) {bar}")
    
    # Top concepts by score
    top_concepts = sorted(concepts, key=lambda x: x.get('score', 0), reverse=True)[:10]
    safe_print(f"\n[TOP] Top 10 Concepts by Score:")
    for i, c in enumerate(top_concepts):
        ctype = c.get('metadata', {}).get('concept_type', 'untyped')
        safe_print(f"  {i+1:>2}. {c['label']:<30} {c.get('score', 0):.3f} [{ctype}]")
    
    # Cluster analysis
    word_freq = Counter()
    for c in concepts:
        words = c['label'].lower().split()
        for word in words:
            if len(word) > 3:
                word_freq[word] += 1
    
    safe_print(f"\n[CLUSTERS] Most Common Terms:")
    for word, count in word_freq.most_common(10):
        safe_print(f"  '{word}': {count} occurrences")
    
    # Normalization status
    normalized_count = sum(1 for c in concepts if 'normalized_label' in c.get('metadata', {}))
    if normalized_count > 0:
        safe_print(f"\n[NORMALIZATION] Synonym Normalization:")
        safe_print(f"  Concepts normalized: {normalized_count}/{len(concepts)} ({normalized_count/len(concepts)*100:.1f}%)")
    
    # Health indicators
    safe_print(f"\n[HEALTH] Vault Health Indicators:")
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
    
    safe_print(f"  Health Score: {health_score}/100")
    if issues:
        safe_print("  Issues:")
        for issue in issues:
            safe_print(f"    ! {issue}")
    else:
        safe_print("  OK All indicators healthy!")
    
    # Recent additions (if timestamps available)
    concepts_with_time = [c for c in concepts if 'created_at' in c.get('metadata', {})]
    if concepts_with_time:
        recent = sorted(concepts_with_time, 
                       key=lambda x: x['metadata']['created_at'], 
                       reverse=True)[:5]
        safe_print(f"\n[RECENT] Recently Added:")
        for c in recent:
            safe_print(f"  - {c['label']} ({c['metadata']['created_at'][:10]})")


if __name__ == "__main__":
    try:
        analyze_vault()
        
        # Check for additional reports
        if Path("soft_duplicates.json").exists():
            safe_print("\n[INFO] Additional reports available:")
            safe_print("  - soft_duplicates.json (run: python soft_dedupe.py)")
        
        if Path("concept_graph.json").exists():
            safe_print("  - concept_graph.json (visualization data)")
        
        audit_dir = Path("data/memory_vault/audits")
        if audit_dir.exists() and list(audit_dir.glob("*.json")):
            safe_print("  - Audit reports in data/memory_vault/audits/")
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        import traceback
        traceback.print_exc()
