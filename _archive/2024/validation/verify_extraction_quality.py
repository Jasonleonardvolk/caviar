#!/usr/bin/env python3
"""
Verify extraction quality and deduplication effectiveness
"""

import json
from pathlib import Path
import logging
from collections import Counter, defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_extraction_quality():
    """Analyze the quality of concept extraction and relationships"""
    
    # Check if we have stored memories
    vault_path = Path("data/memory_vault/memories")
    if not vault_path.exists():
        logger.error("❌ No memory vault found")
        return
    
    memory_files = list(vault_path.glob("*.json"))
    logger.info(f"📁 Found {len(memory_files)} memories in vault")
    
    # Analyze concepts
    concepts = []
    relationships_by_type = defaultdict(int)
    concept_methods = Counter()
    concept_scores = []
    
    for memory_file in memory_files:
        with open(memory_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        content = data.get('content', {})
        concepts.append(content)
        
        # Track methods
        method = content.get('method', 'unknown')
        for m in method.split('+'):
            concept_methods[m.strip()] += 1
        
        # Track scores
        score = content.get('score')
        if score:
            concept_scores.append(score)
        
        # Analyze relationships
        for rel in content.get('relationships', []):
            rel_type = rel.get('type', 'unknown')
            relationships_by_type[rel_type] += 1
    
    # Report findings
    logger.info("\n" + "="*60)
    logger.info("📊 EXTRACTION QUALITY REPORT")
    logger.info("="*60)
    
    logger.info(f"\n📈 Concept Statistics:")
    logger.info(f"  Total concepts: {len(concepts)}")
    logger.info(f"  Average score: {sum(concept_scores)/len(concept_scores):.3f}" if concept_scores else "  No scores")
    logger.info(f"  Score range: {min(concept_scores):.3f} - {max(concept_scores):.3f}" if concept_scores else "")
    
    logger.info(f"\n🛠️ Extraction Methods:")
    for method, count in concept_methods.most_common():
        logger.info(f"  {method}: {count}")
    
    logger.info(f"\n🔗 Relationship Types:")
    total_rels = sum(relationships_by_type.values())
    logger.info(f"  Total relationships: {total_rels}")
    for rel_type, count in sorted(relationships_by_type.items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = (count / total_rels * 100) if total_rels > 0 else 0
        logger.info(f"  {rel_type}: {count} ({percentage:.1f}%)")
    
    # Find concepts with most relationships
    concepts_by_rel_count = sorted(
        concepts, 
        key=lambda c: len(c.get('relationships', [])), 
        reverse=True
    )
    
    logger.info(f"\n🏆 Most Connected Concepts:")
    for concept in concepts_by_rel_count[:10]:
        rel_count = len(concept.get('relationships', []))
        if rel_count > 0:
            logger.info(f"  {concept['label']}: {rel_count} relationships")
            # Show first few relationships
            for rel in concept['relationships'][:3]:
                logger.info(f"    → {rel['type']}: {rel['target']}")
    
    # Check for duplicates that might have been missed
    labels = [c['label'].lower() for c in concepts]
    label_counts = Counter(labels)
    duplicates = [(label, count) for label, count in label_counts.items() if count > 1]
    
    if duplicates:
        logger.info(f"\n⚠️ Potential Duplicates Found:")
        for label, count in duplicates:
            logger.info(f"  '{label}': {count} instances")
    else:
        logger.info(f"\n✅ No duplicate concepts found - deduplication working well!")
    
    # Analyze semantic clusters
    logger.info(f"\n🎯 Concept Clusters (by common words):")
    word_groups = defaultdict(list)
    for concept in concepts:
        label = concept['label']
        words = label.lower().split()
        for word in words:
            if len(word) > 3:  # Skip short words
                word_groups[word].append(label)
    
    # Show largest clusters
    clusters = sorted(word_groups.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for word, group in clusters:
        if len(group) > 1:
            logger.info(f"  '{word}': {len(group)} concepts")
            for concept in group[:5]:
                logger.info(f"    - {concept}")
    
    return concepts


def generate_graph_data(concepts):
    """Generate data for relationship graph visualization"""
    
    # Create nodes and edges for graph
    nodes = []
    edges = []
    
    # Create label to ID mapping
    label_to_id = {c['label']: c['id'] for c in concepts}
    
    for concept in concepts:
        # Add node
        nodes.append({
            'id': concept['id'],
            'label': concept['label'],
            'score': concept.get('score', 0.5),
            'method': concept.get('method', 'unknown')
        })
        
        # Add edges for relationships
        for rel in concept.get('relationships', []):
            target_label = rel.get('target', '')
            # Try to find target in our concepts
            target_id = label_to_id.get(target_label)
            
            edges.append({
                'source': concept['id'],
                'target': target_id or f"external_{target_label}",
                'type': rel.get('type', 'related'),
                'verb': rel.get('verb', '')
            })
    
    graph_data = {
        'nodes': nodes,
        'edges': edges,
        'stats': {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'connected_nodes': len(set(e['source'] for e in edges))
        }
    }
    
    # Save graph data
    with open('concept_graph.json', 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2)
    
    logger.info(f"\n📊 Graph data saved to concept_graph.json")
    logger.info(f"  Nodes: {len(nodes)}")
    logger.info(f"  Edges: {len(edges)}")


if __name__ == "__main__":
    logger.info("🔍 Analyzing extraction quality...")
    concepts = analyze_extraction_quality()
    
    if concepts:
        logger.info("\n📊 Generating graph visualization data...")
        generate_graph_data(concepts)
        
        logger.info("\n✅ Quality analysis complete!")
        logger.info("\n💡 Next steps:")
        logger.info("  1. Review concept_graph.json for visualization")
        logger.info("  2. Check for any unexpected duplicates")
        logger.info("  3. Verify relationship quality")
