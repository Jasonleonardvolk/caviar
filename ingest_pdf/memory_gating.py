"""memory_gating.py - Implements spectral entropy thresholds and redundancy detection.

This module provides mechanisms to prevent memory bloat in the concept graph by:
1. Calculating spectral entropy of concept clusters
2. Detecting and merging redundant concepts
3. Pruning low-impact concepts based on usage statistics

These methods support ALAN's "No Memory Bloat" commitment.
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Any, Optional
from collections import Counter
import logging

try:
    # Try absolute import first
    from models import ConceptTuple
except ImportError:
    # Fallback to relative import
    try:
        # Try absolute import first
        from models import ConceptTuple
    except ImportError:
        # Fallback to relative import
        from .models import ConceptTuple

# Configure logger
logger = logging.getLogger("memory_gating")

def calculate_spectral_entropy(vectors: np.ndarray, eps: float = 1e-10) -> float:
    """
    Calculate the normalized spectral entropy of a set of vectors.
    
    Spectral entropy measures the complexity/information content of a set of vectors
    by analyzing the eigenvalue distribution of their covariance matrix.
    
    Args:
        vectors: Array of shape (n_vectors, vector_dim)
        eps: Small epsilon to prevent log(0)
        
    Returns:
        Normalized entropy in [0, 1]
    """
    if len(vectors) < 2:
        return 0.0
    
    # Calculate covariance matrix
    vectors = np.array(vectors)
    cov = vectors @ vectors.T
    
    # Calculate eigenvalues
    try:
        eigvals = np.linalg.eigvalsh(cov)
        # Filter out very small or negative eigenvalues
        eigvals = eigvals[eigvals > eps]
        
        if len(eigvals) == 0:
            return 0.0
        
        # Normalize eigenvalues to get "probabilities"
        p = eigvals / eigvals.sum()
        
        # Calculate Shannon entropy
        entropy = -np.sum(p * np.log2(p + eps))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(p))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    except np.linalg.LinAlgError:
        logger.warning("Error computing spectral entropy: Covariance matrix is singular")
        return 0.0

def jaccard_similarity(set1: Set, set2: Set) -> float:
    """
    Calculate Jaccard similarity between two sets.
    
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    """
    if not set1 or not set2:
        return 0.0
        
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return float(np.dot(v1, v2) / (norm1 * norm2))

def concept_redundancy(
    concept1: ConceptTuple, 
    concept2: ConceptTuple, 
    text_weight: float = 0.3,
    embedding_weight: float = 0.5,
    cluster_weight: float = 0.2
) -> float:
    """
    Calculate redundancy score between two concepts.
    
    Combines multiple similarity metrics:
    1. Jaccard similarity of cluster members
    2. Cosine similarity of concept embeddings
    3. Text similarity of concept names
    
    Args:
        concept1, concept2: ConceptTuple instances to compare
        text_weight: Weight for name similarity
        embedding_weight: Weight for embedding similarity
        cluster_weight: Weight for cluster membership similarity
        
    Returns:
        Redundancy score in [0, 1], where 1 = completely redundant
    """
    # Embedding similarity (vector space)
    emb_sim = cosine_similarity(concept1.embedding, concept2.embedding)
    
    # Cluster membership similarity
    cluster_sim = jaccard_similarity(
        set(concept1.cluster_members), 
        set(concept2.cluster_members)
    )
    
    # Text similarity (concept names)
    # Simple word overlap for demonstration
    words1 = set(concept1.name.lower().split())
    words2 = set(concept2.name.lower().split())
    text_sim = jaccard_similarity(words1, words2)
    
    # Weighted combination
    redundancy = (
        text_weight * text_sim +
        embedding_weight * emb_sim +
        cluster_weight * cluster_sim
    )
    
    return redundancy

def detect_redundant_concepts(
    concepts: List[ConceptTuple], 
    threshold: float = 0.8
) -> List[Tuple[int, int, float]]:
    """
    Detect pairs of redundant concepts based on similarity.
    
    Args:
        concepts: List of ConceptTuple objects
        threshold: Redundancy threshold above which concepts are considered redundant
        
    Returns:
        List of tuples (idx1, idx2, redundancy) for redundant concept pairs
    """
    redundant_pairs = []
    
    for i in range(len(concepts)):
        for j in range(i+1, len(concepts)):
            redundancy = concept_redundancy(concepts[i], concepts[j])
            
            if redundancy >= threshold:
                redundant_pairs.append((i, j, redundancy))
                logger.info(f"Redundant concepts detected: '{concepts[i].name}' and '{concepts[j].name}'"
                           f" (score: {redundancy:.2f})")
    
    # Sort by redundancy (highest first)
    redundant_pairs.sort(key=lambda x: x[2], reverse=True)
    return redundant_pairs

def merge_redundant_concepts(
    concepts: List[ConceptTuple],
    redundant_pairs: List[Tuple[int, int, float]]
) -> List[ConceptTuple]:
    """
    Merge redundant concepts to reduce memory bloat.
    
    Args:
        concepts: List of ConceptTuple objects
        redundant_pairs: List of (idx1, idx2, score) tuples for redundant pairs
        
    Returns:
        New list with merged concepts
    """
    if not redundant_pairs:
        return concepts
        
    # Track indices to merge and remove
    to_remove = set()
    
    # Create a copy of the concepts list
    result = list(concepts)
    
    for idx1, idx2, _ in redundant_pairs:
        # Skip if either concept has already been merged
        if idx1 in to_remove or idx2 in to_remove:
            continue
            
        c1 = result[idx1]
        c2 = result[idx2]
        
        # Create merged concept
        merged_name = f"{c1.name} / {c2.name}"
        merged_embedding = (c1.embedding + c2.embedding) / 2
        merged_members = sorted(list(set(c1.cluster_members + c2.cluster_members)))
        merged_context = c1.context if len(c1.context) >= len(c2.context) else c2.context
        
        # Use average weights for scores
        merged_resonance = (c1.resonance_score + c2.resonance_score) / 2
        merged_centrality = (c1.narrative_centrality + c2.narrative_centrality) / 2
        merged_predictability = (c1.predictability_score + c2.predictability_score) / 2
        
        # Combine spectral lineage data
        merged_lineage = c1.spectral_lineage + c2.spectral_lineage
        
        # Combine source provenance
        merged_provenance = {**c1.source_provenance, **c2.source_provenance}
        
        # Create merged ConceptTuple
        merged_concept = ConceptTuple(
            name=merged_name,
            embedding=merged_embedding,
            context=merged_context,
            passage_embedding=c1.passage_embedding,  # Use one of the passage embeddings
            cluster_members=merged_members,
            resonance_score=merged_resonance,
            narrative_centrality=merged_centrality,
            predictability_score=merged_predictability,
            # New fields (will generate eigenfunction_id automatically)
            source_provenance=merged_provenance,
            spectral_lineage=merged_lineage,
            cluster_coherence=(c1.cluster_coherence + c2.cluster_coherence) / 2
        )
        
        # Replace the first concept with merged concept
        result[idx1] = merged_concept
        # Mark second concept for removal
        to_remove.add(idx2)
        
        logger.info(f"Merged concepts: '{c1.name}' + '{c2.name}' → '{merged_name}'")
    
    # Remove concepts marked for deletion (in reverse order to maintain indices)
    for idx in sorted(to_remove, reverse=True):
        del result[idx]
    
    return result

def prune_low_entropy_concepts(
    concepts: List[ConceptTuple],
    min_entropy: float = 0.2,
    min_resonance: float = 0.3
) -> List[ConceptTuple]:
    """
    Prune concepts with low spectral entropy (carrying little unique information).
    
    Args:
        concepts: List of ConceptTuple objects
        min_entropy: Minimum required spectral entropy
        min_resonance: Minimum required resonance score
        
    Returns:
        Filtered list of concepts
    """
    result = []
    pruned_count = 0
    
    for concept in concepts:
        # Skip if no embedding is available
        if concept.embedding is None or len(concept.cluster_members) < 2:
            result.append(concept)
            continue
            
        # Collect embeddings for entropy calculation
        # In a real implementation, we'd use actual block embeddings
        # Here we simulate by creating variants of the concept embedding
        if hasattr(concept, 'passage_embedding') and concept.passage_embedding is not None:
            # Use concept embedding and passage embedding
            embeddings = np.array([concept.embedding, concept.passage_embedding])
        else:
            # Create synthetic block embeddings for demonstration
            # (in a real implementation these would come from actual blocks)
            noise = np.random.normal(0, 0.1, size=concept.embedding.shape)
            embeddings = np.array([
                concept.embedding,
                concept.embedding + noise,
                concept.embedding - noise
            ])
        
        # Calculate spectral entropy
        entropy = calculate_spectral_entropy(embeddings)
        
        # Keep concept if it has high enough entropy OR high resonance
        if entropy >= min_entropy or concept.resonance_score >= min_resonance:
            result.append(concept)
        else:
            pruned_count += 1
            logger.info(f"Pruned low-entropy concept: '{concept.name}' "
                       f"(entropy: {entropy:.2f}, resonance: {concept.resonance_score:.2f})")
    
    logger.info(f"Pruned {pruned_count}/{len(concepts)} concepts with low entropy/resonance")
    return result

def apply_memory_gating(concepts: List[ConceptTuple]) -> List[ConceptTuple]:
    """
    Apply full memory gating pipeline to prevent memory bloat:
    1. Detect and merge redundant concepts
    2. Prune low entropy concepts
    
    Args:
        concepts: List of ConceptTuple objects
        
    Returns:
        Filtered and merged concept list
    """
    if not concepts:
        return []
        
    logger.info(f"Applying memory gating to {len(concepts)} concepts")
    
    # Step 1: Detect redundant concepts
    redundant_pairs = detect_redundant_concepts(concepts)
    
    # Step 2: Merge redundant concepts
    merged_concepts = merge_redundant_concepts(concepts, redundant_pairs)
    
    # Step 3: Prune low entropy concepts
    pruned_concepts = prune_low_entropy_concepts(merged_concepts)
    
    # Log summary
    reduction = len(concepts) - len(pruned_concepts)
    if reduction > 0:
        logger.info(f"Memory gating reduced concept count by {reduction} "
                   f"({100*reduction/len(concepts):.1f}%)")
    
    return pruned_concepts
