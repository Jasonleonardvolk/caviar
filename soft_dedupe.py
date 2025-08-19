#!/usr/bin/env python3
"""
Soft Duplicate Detection for Memory Vault
Uses semantic similarity to find near-duplicate concepts
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available - soft deduplication will use basic string matching")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SoftDeduplicator:
    """Detect and report soft duplicates in memory vault"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.nlp = None
        
        if SPACY_AVAILABLE:
            try:
                # Try medium model first for better vectors
                self.nlp = spacy.load("en_core_web_md")
                logger.info("✅ Loaded en_core_web_md for similarity detection")
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.warning("⚠️ Using en_core_web_sm (limited similarity)")
                except Exception as e:
                    logger.error(f"❌ No spaCy model available: {e}")
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if not self.nlp:
            # Fallback to Jaccard similarity
            return self._jaccard_similarity(text1, text2)
        
        try:
            doc1 = self.nlp(text1)
            doc2 = self.nlp(text2)
            
            if doc1.has_vector and doc2.has_vector:
                return doc1.similarity(doc2)
            else:
                # Fallback for models without vectors
                return self._token_overlap_similarity(doc1, doc2)
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity for fallback"""
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _token_overlap_similarity(self, doc1, doc2) -> float:
        """Token-based similarity for spaCy docs without vectors"""
        tokens1 = set(t.lemma_.lower() for t in doc1 if t.is_alpha and not t.is_stop)
        tokens2 = set(t.lemma_.lower() for t in doc2 if t.is_alpha and not t.is_stop)
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        smaller = min(len(tokens1), len(tokens2))
        
        # Use smaller set as denominator for better detection of substrings
        return intersection / smaller if smaller > 0 else 0.0
    
    def find_duplicates(self, concepts: List[Dict]) -> List[Dict]:
        """Find soft duplicates in concept list"""
        duplicates = []
        n = len(concepts)
        
        logger.info(f"🔍 Checking {n} concepts for soft duplicates...")
        
        # Build similarity matrix
        similarities = []
        
        for i in range(n):
            for j in range(i + 1, n):
                label1 = concepts[i].get('label', '')
                label2 = concepts[j].get('label', '')
                
                if not label1 or not label2:
                    continue
                
                similarity = self.calculate_similarity(label1, label2)
                
                if similarity >= self.similarity_threshold:
                    similarities.append({
                        'concept1': {
                            'id': concepts[i].get('id', f'idx_{i}'),
                            'label': label1,
                            'score': concepts[i].get('score', 0),
                            'method': concepts[i].get('method', 'unknown')
                        },
                        'concept2': {
                            'id': concepts[j].get('id', f'idx_{j}'),
                            'label': label2,
                            'score': concepts[j].get('score', 0),
                            'method': concepts[j].get('method', 'unknown')
                        },
                        'similarity': similarity,
                        'recommendation': 'merge' if similarity > 0.95 else 'review'
                    })
        
        # Group by concept to find clusters
        clusters = self._find_clusters(similarities)
        
        return {
            'pairs': similarities,
            'clusters': clusters,
            'stats': {
                'total_concepts': n,
                'duplicate_pairs': len(similarities),
                'clusters_found': len(clusters),
                'threshold_used': self.similarity_threshold
            }
        }
    
    def _find_clusters(self, similarities: List[Dict]) -> List[Dict]:
        """Group similar concepts into clusters"""
        if not similarities:
            return []
        
        # Build adjacency list
        graph = {}
        for sim in similarities:
            id1 = sim['concept1']['id']
            id2 = sim['concept2']['id']
            
            if id1 not in graph:
                graph[id1] = set()
            if id2 not in graph:
                graph[id2] = set()
            
            graph[id1].add(id2)
            graph[id2].add(id1)
        
        # Find connected components
        visited = set()
        clusters = []
        
        for node in graph:
            if node not in visited:
                cluster = []
                stack = [node]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        cluster.append(current)
                        stack.extend(graph[current] - visited)
                
                if len(cluster) > 1:
                    # Get full concept info for cluster
                    cluster_info = []
                    for pair in similarities:
                        if pair['concept1']['id'] in cluster:
                            cluster_info.append(pair['concept1'])
                        if pair['concept2']['id'] in cluster:
                            cluster_info.append(pair['concept2'])
                    
                    # Deduplicate cluster info
                    seen_ids = set()
                    unique_cluster = []
                    for concept in cluster_info:
                        if concept['id'] not in seen_ids:
                            seen_ids.add(concept['id'])
                            unique_cluster.append(concept)
                    
                    # Sort by score to identify primary concept
                    unique_cluster.sort(key=lambda x: x.get('score', 0), reverse=True)
                    
                    clusters.append({
                        'size': len(unique_cluster),
                        'primary': unique_cluster[0],
                        'members': unique_cluster,
                        'recommendation': f"Keep '{unique_cluster[0]['label']}' (highest score)"
                    })
        
        return sorted(clusters, key=lambda x: x['size'], reverse=True)


def analyze_vault_duplicates(vault_path: Path, output_path: Path = None):
    """Main function to analyze vault for duplicates"""
    
    # Load all concepts from vault
    concepts = []
    memory_files = list(vault_path.glob("*.json"))
    
    logger.info(f"📁 Loading {len(memory_files)} memory files...")
    
    for file in memory_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            content = data.get('content', {})
            if isinstance(content, dict) and 'label' in content:
                content['id'] = content.get('id', file.stem)
                concepts.append(content)
        except Exception as e:
            logger.warning(f"Failed to load {file.name}: {e}")
    
    logger.info(f"✅ Loaded {len(concepts)} valid concepts")
    
    # Find duplicates
    deduplicator = SoftDeduplicator(similarity_threshold=0.85)
    results = deduplicator.find_duplicates(concepts)
    
    # Report findings
    logger.info("\n" + "="*60)
    logger.info("🧬 SOFT DUPLICATE ANALYSIS")
    logger.info("="*60)
    logger.info(f"Total concepts: {results['stats']['total_concepts']}")
    logger.info(f"Duplicate pairs found: {results['stats']['duplicate_pairs']}")
    logger.info(f"Duplicate clusters: {results['stats']['clusters_found']}")
    logger.info(f"Similarity threshold: {results['stats']['threshold_used']}")
    
    if results['clusters']:
        logger.info("\n📊 Duplicate Clusters:")
        for i, cluster in enumerate(results['clusters'][:5]):
            logger.info(f"\nCluster {i+1} ({cluster['size']} concepts):")
            logger.info(f"  Primary: '{cluster['primary']['label']}' (score: {cluster['primary']['score']:.3f})")
            logger.info("  Members:")
            for member in cluster['members']:
                logger.info(f"    - '{member['label']}' (score: {member.get('score', 0):.3f})")
            logger.info(f"  💡 {cluster['recommendation']}")
    
    # Save results
    if output_path is None:
        output_path = Path("soft_duplicates.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n📄 Detailed results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    vault_path = Path("data/memory_vault/memories")
    
    if not vault_path.exists():
        logger.error("❌ Memory vault not found")
    else:
        results = analyze_vault_duplicates(vault_path)
        
        if results['stats']['duplicate_pairs'] > 0:
            logger.info("\n💡 Recommendations:")
            logger.info("  1. Review soft_duplicates.json for detailed pairs")
            logger.info("  2. Consider merging high-similarity pairs (>0.95)")
            logger.info("  3. Update extraction pipeline to prevent future duplicates")
