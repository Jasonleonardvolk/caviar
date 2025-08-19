import numpy as np
from typing import List, Dict, Any
from collections import Counter

# Import threshold configuration
try:
    try:
        # Try absolute import first
        from threshold_config import MIN_CONFIDENCE, get_threshold_for_media_type, get_adaptive_threshold
    except ImportError:
        # Fallback to relative import
        from .threshold_config import MIN_CONFIDENCE, get_threshold_for_media_type, get_adaptive_threshold
except ImportError:
    # Fallback if threshold_config is not available
    MIN_CONFIDENCE = 0.5
    def get_threshold_for_media_type(media_type: str, base_threshold: float = None) -> float:
        return base_threshold or MIN_CONFIDENCE
    def get_adaptive_threshold(content_length: int, media_type: str = "pdf") -> float:
        return MIN_CONFIDENCE

def resonance_score(cluster_indices: List[int], emb: np.ndarray) -> float:
    # Simple autocorrelation as resonance proxy
    if len(cluster_indices) < 2:
        return 0.0
    signal = emb[cluster_indices].mean(axis=0)
    norm = np.linalg.norm(signal) + 1e-8
    ac = np.correlate(signal, signal, mode='full')
    center = len(ac) // 2
    # Use off-center peak as resonance
    if len(ac) > 2:
        resonance = float(np.max(ac[center+1:]) / (norm**2))
    else:
        resonance = 0.0
    return resonance

def narrative_centrality(cluster_indices: List[int], adjacency: np.ndarray) -> float:
    # Degree centrality in cluster graph
    if len(cluster_indices) < 1:
        return 0.0
    sub_adj = adjacency[np.ix_(cluster_indices, cluster_indices)]
    deg = sub_adj.sum(axis=1)
    return float(np.mean(deg))

def build_cluster_adjacency(labels: List[int], emb: np.ndarray) -> np.ndarray:
    # Build adjacency based on cosine similarity between clusters
    n = len(labels)
    unique = sorted(set(labels))
    cluster_means = [emb[[i for i, l in enumerate(labels) if l == cid]].mean(axis=0) for cid in unique]
    m = len(cluster_means)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                continue
            sim = float(emb[i] @ emb[j] / ((np.linalg.norm(emb[i])*np.linalg.norm(emb[j]))+1e-8))
            adj[i, j] = sim
    return adj

def score_clusters(labels: List[int], emb: np.ndarray) -> List[int]:
    scores = {}
    adj = build_cluster_adjacency(labels, emb)
    for cid in set(labels):
        mem = [i for i, l in enumerate(labels) if l == cid]
        if len(mem) < 1:
            continue
        res = resonance_score(mem, emb)
        cent = narrative_centrality(mem, adj)
        # Weighted score: resonance + centrality
        scores[cid] = 0.6 * res + 0.4 * cent
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]

def filter_concepts(
    concepts: List[Dict[str, Any]], 
    threshold: float = None,
    media_type: str = "pdf",
    content_length: int = None
) -> List[Dict[str, Any]]:
    """
    Filter concepts based on confidence threshold (addresses Issue #2 - overzealous filtering).
    
    Args:
        concepts: List of concept dictionaries with confidence scores
        threshold: Confidence threshold (defaults to MIN_CONFIDENCE or adaptive)
        media_type: Type of media for threshold adjustment
        content_length: Length of content for adaptive thresholding
        
    Returns:
        List of concepts that meet the confidence threshold
    """
    if not concepts:
        return []
    
    # Determine threshold to use
    if threshold is None:
        if content_length is not None:
            threshold = get_adaptive_threshold(content_length, media_type)
        else:
            threshold = get_threshold_for_media_type(media_type)
    
    # Filter concepts by confidence
    filtered = []
    for concept in concepts:
        if not isinstance(concept, dict):
            continue
            
        confidence = concept.get("confidence", 0.0)
        if isinstance(confidence, (int, float)) and confidence >= threshold:
            filtered.append(concept)
    
    return filtered

def apply_confidence_fallback(
    filtered_concepts: List[Dict[str, Any]], 
    all_concepts: List[Dict[str, Any]], 
    min_count: int = 3
) -> List[Dict[str, Any]]:
    """
    Apply fallback logic to ensure minimum concept count (addresses Issue #2 - empty results).
    
    Args:
        filtered_concepts: Concepts that passed confidence filtering
        all_concepts: All candidate concepts
        min_count: Minimum number of concepts to return
        
    Returns:
        Concepts with fallback applied if needed
    """
    if len(filtered_concepts) >= min_count or len(all_concepts) <= min_count:
        return filtered_concepts
    
    # Sort all concepts by confidence and take top min_count
    sorted_concepts = sorted(
        all_concepts, 
        key=lambda c: c.get("confidence", 0.0), 
        reverse=True
    )
    
    return sorted_concepts[:min_count]

def calculate_concept_confidence(
    cluster_indices: List[int],
    embeddings: np.ndarray,
    labels: List[int],
    method: str = "combined"
) -> float:
    """
    Calculate confidence score for a concept cluster.
    
    Args:
        cluster_indices: Indices of items in the cluster
        embeddings: Embedding matrix
        labels: Cluster labels for all items
        method: Method for confidence calculation
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not cluster_indices:
        return 0.0
    
    if method == "resonance":
        return min(1.0, resonance_score(cluster_indices, embeddings))
    elif method == "centrality":
        adj = build_cluster_adjacency(labels, embeddings)
        return min(1.0, narrative_centrality(cluster_indices, adj) / 10.0)  # Normalize
    elif method == "cohesion":
        # Internal cluster cohesion
        if len(cluster_indices) < 2:
            return 0.5  # Single item clusters get medium confidence
        
        cluster_embs = embeddings[cluster_indices]
        pairwise_sims = []
        for i in range(len(cluster_embs)):
            for j in range(i + 1, len(cluster_embs)):
                sim = np.dot(cluster_embs[i], cluster_embs[j]) / (
                    np.linalg.norm(cluster_embs[i]) * np.linalg.norm(cluster_embs[j]) + 1e-8
                )
                pairwise_sims.append(sim)
        
        return float(np.mean(pairwise_sims)) if pairwise_sims else 0.5
    else:  # combined
        res = resonance_score(cluster_indices, embeddings)
        adj = build_cluster_adjacency(labels, embeddings)
        cent = narrative_centrality(cluster_indices, adj) / 10.0  # Normalize
        
        # Cohesion component
        cohesion = 0.5
        if len(cluster_indices) >= 2:
            cluster_embs = embeddings[cluster_indices]
            pairwise_sims = []
            for i in range(len(cluster_embs)):
                for j in range(i + 1, len(cluster_embs)):
                    sim = np.dot(cluster_embs[i], cluster_embs[j]) / (
                        np.linalg.norm(cluster_embs[i]) * np.linalg.norm(cluster_embs[j]) + 1e-8
                    )
                    pairwise_sims.append(sim)
            cohesion = float(np.mean(pairwise_sims)) if pairwise_sims else 0.5
        
        # Weighted combination
        combined = 0.4 * res + 0.3 * cent + 0.3 * cohesion
        return min(1.0, max(0.0, combined))

def rank_concepts_by_quality(
    concepts: List[Dict[str, Any]], 
    reverse: bool = True
) -> List[Dict[str, Any]]:
    """
    Rank concepts by overall quality score.
    
    Args:
        concepts: List of concept dictionaries
        reverse: Whether to sort in descending order (highest quality first)
        
    Returns:
        Sorted list of concepts
    """
    def quality_score(concept: Dict[str, Any]) -> float:
        confidence = concept.get("confidence", 0.0)
        resonance = concept.get("resonance_score", 0.0)
        centrality = concept.get("narrative_centrality", 0.0)
        
        # Weighted quality score
        return 0.5 * confidence + 0.3 * resonance + 0.2 * centrality
    
    return sorted(concepts, key=quality_score, reverse=reverse)
