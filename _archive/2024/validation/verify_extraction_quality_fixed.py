#!/usr/bin/env python3
"""
Verify extraction quality and deduplication effectiveness - Fixed version
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
        return []
    
    memory_files = list(vault_path.glob("*.json"))
    logger.info(f"📁 Found {len(memory_files)} memory files in vault")
    
    # Analyze concepts
    concepts = []
    relationships_by_type = defaultdict(int)
    concept_methods = Counter()
    concept_scores = []
    failed_files = []
    
    for memory_file in memory_files:
        try:
            # Check file size first
            if memory_file.stat().st_size == 0:
                logger.warning(f"⚠️ Empty file: {memory_file.name}")
                failed_files.append(memory_file.name)
                continue
                
            with open(memory_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
                if not file_content.strip():
                    logger.warning(f"⚠️ Empty content in: {memory_file.name}")
                    failed_files.append(memory_file.name)
                    continue
                    
                # Try to parse JSON
                data = json.loads(file_content)
            
            content = data.get('content', {})
            if not content:
                logger.warning(f"⚠️ No content in: {memory_file.name}")
                continue
                
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
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON error in {memory_file.name}: {e}")
            failed_files.append(memory_file.name)
        except Exception as e:
            logger.error(f"❌ Error processing {memory_file.name}: {e}")
            failed_files.append(memory_file.name)
    
    # Report findings
    logger.info("\n" + "="*60)
    logger.info("📊 EXTRACTION QUALITY REPORT")
    logger.info("="*60)
    
    if failed_files:
        logger.warning(f"\n⚠️ Failed to process {len(failed_files)} files:")
        for fname in failed_files[:5]:
            logger.warning(f"  - {fname}")
    
    logger.info(f"\n📈 Concept Statistics:")
    logger.info(f"  Total concepts loaded: {len(concepts)}")
    logger.info(f"  Total files: {len(memory_files)}")
    logger.info(f"  Successfully processed: {len(concepts)}")
    
    if concept_scores:
        logger.info(f"  Average score: {sum(concept_scores)/len(concept_scores):.3f}")
        logger.info(f"  Score range: {min(concept_scores):.3f} - {max(concept_scores):.3f}")
    
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
    valid_concepts = [c for c in concepts if isinstance(c, dict)]  # Filter out any non-dict items
    
    concepts_by_rel_count = sorted(
        valid_concepts, 
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
    # Only process concepts that have labels
    labeled_concepts = [c for c in concepts if isinstance(c, dict) and 'label' in c]
    labels = [c['label'].lower() for c in labeled_concepts]
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
            unique_group = list(set(group))[:5]
            for concept in unique_group:
                logger.info(f"    - {concept}")
    
    return concepts


def generate_graph_data(concepts):
    """Generate data for relationship graph visualization"""
    
    if not concepts:
        logger.warning("⚠️ No concepts to generate graph from")
        return
    
    # Create nodes and edges for graph
    nodes = []
    edges = []
    
    # Create label to ID mapping
    label_to_id = {}
    for concept in concepts:
        cid = concept.get('id', f"id_{len(label_to_id)}")
        label_to_id[concept['label']] = cid
    
    for concept in concepts:
        cid = concept.get('id', label_to_id.get(concept['label'], f"id_{len(nodes)}"))
        
        # Add node
        nodes.append({
            'id': cid,
            'label': concept['label'],
            'score': concept.get('score', 0.5),
            'method': concept.get('method', 'unknown'),
            'relationships_count': len(concept.get('relationships', []))
        })
        
        # Add edges for relationships
        for rel in concept.get('relationships', []):
            target_label = rel.get('target', '')
            # Try to find target in our concepts
            target_id = label_to_id.get(target_label)
            
            edges.append({
                'source': cid,
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
    logger.info(f"  Connected nodes: {graph_data['stats']['connected_nodes']}")


def check_vault_integrity():
    """Quick check of vault file integrity"""
    vault_path = Path("data/memory_vault/memories")
    if not vault_path.exists():
        return
    
    logger.info("\n🔍 Checking vault integrity...")
    
    memory_files = list(vault_path.glob("*.json"))
    empty_files = []
    
    for mf in memory_files:
        if mf.stat().st_size == 0:
            empty_files.append(mf)
    
    if empty_files:
        logger.warning(f"⚠️ Found {len(empty_files)} empty files")
        for ef in empty_files:
            logger.info(f"  Removing empty file: {ef.name}")
            ef.unlink()  # Delete empty file
    else:
        logger.info("✅ All files have content")


if __name__ == "__main__":
    # First check vault integrity
    check_vault_integrity()
    
    # Then analyze
    logger.info("\n🔍 Analyzing extraction quality...")
    concepts = analyze_extraction_quality()
    
    if concepts:
        logger.info("\n📊 Generating graph visualization data...")
        generate_graph_data(concepts)
        
        logger.info("\n✅ Quality analysis complete!")
        logger.info("\n💡 Next steps:")
        logger.info("  1. Review concept_graph.json for visualization")
        logger.info("  2. Install en_core_web_md for better similarity: python install_spacy_medium.py")
        logger.info("  3. Check extraction_summary.json for pipeline stats")
