"""enhanced_memory_gating.py - Implements advanced spectral entropy-based memory management.

This module provides enhanced mechanisms to prevent memory bloat in the concept graph by:
1. Implementing configurable spectral entropy thresholds
2. Time-based resonance decay for concept pruning
3. O-information based redundancy detection for multi-concept relationships
4. Adaptive threshold adjustment based on concept density

These methods support ALAN's "No Memory Bloat" commitment while preserving
conceptual integrity and coherence.
"""

import numpy as np
import time
from datetime import datetime
from typing import List, Dict, Tuple, Set, Any, Optional, Union
from collections import Counter, defaultdict
import logging
import uuid
import math
from dataclasses import dataclass, field

try:
    # Try absolute import first
    from models import ConceptTuple
except ImportError:
    # Fallback to relative import
    from .models import ConceptTuple
try:
    # Try absolute import first
    from memory_gating import calculate_spectral_entropy, cosine_similarity, jaccard_similarity
except ImportError:
    # Fallback to relative import
    from .memory_gating import calculate_spectral_entropy, cosine_similarity, jaccard_similarity

# Configure logger
logger = logging.getLogger("enhanced_memory_gating")

@dataclass
class MemoryGateConfig:
    """Configuration parameters for memory gating algorithms."""
    # Spectral entropy thresholds
    base_entropy_threshold: float = 0.2
    adaptive_entropy_scaling: bool = True
    min_entropy_threshold: float = 0.1
    max_entropy_threshold: float = 0.8
    
    # Resonance decay parameters
    enable_resonance_decay: bool = True
    half_life_days: float = 30.0  # Half-life in days
    min_resonance_keep: float = 0.3
    resonance_decay_factor: float = 0.1
    
    # Redundancy detection
    redundancy_threshold: float = 0.75
    text_similarity_weight: float = 0.3
    embedding_similarity_weight: float = 0.5
    cluster_similarity_weight: float = 0.2
    
    # O-information thresholds
    use_o_information: bool = True
    o_information_threshold: float = 0.3
    
    # Density-based parameters
    concept_density_threshold: int = 1000  # Concepts per dimension
    density_scaling_factor: float = 0.5
    
    # Cache parameters
    similarity_cache_size: int = 10000
    
    # Auto-tuning
    auto_tune_parameters: bool = False
    target_memory_reduction: float = 0.2  # Target 20% reduction
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "base_entropy_threshold": self.base_entropy_threshold,
            "adaptive_entropy_scaling": self.adaptive_entropy_scaling,
            "min_entropy_threshold": self.min_entropy_threshold,
            "max_entropy_threshold": self.max_entropy_threshold,
            "enable_resonance_decay": self.enable_resonance_decay,
            "half_life_days": self.half_life_days,
            "min_resonance_keep": self.min_resonance_keep,
            "resonance_decay_factor": self.resonance_decay_factor,
            "redundancy_threshold": self.redundancy_threshold,
            "text_similarity_weight": self.text_similarity_weight,
            "embedding_similarity_weight": self.embedding_similarity_weight,
            "cluster_similarity_weight": self.cluster_similarity_weight,
            "use_o_information": self.use_o_information,
            "o_information_threshold": self.o_information_threshold,
            "concept_density_threshold": self.concept_density_threshold,
            "density_scaling_factor": self.density_scaling_factor,
            "similarity_cache_size": self.similarity_cache_size,
            "auto_tune_parameters": self.auto_tune_parameters,
            "target_memory_reduction": self.target_memory_reduction
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MemoryGateConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


class SimilarityCache:
    """Cache for similarity computations to avoid redundant calculations."""
    
    def __init__(self, max_size: int = 10000):
        """Initialize cache with maximum size."""
        self.max_size = max_size
        self.cache = {}
        self.access_timestamps = {}
    
    def get_key(self, id1: str, id2: str) -> str:
        """Generate a consistent cache key regardless of order."""
        return f"{min(id1, id2)}:{max(id1, id2)}"
    
    def get(self, id1: str, id2: str) -> Optional[float]:
        """Get cached similarity if available."""
        key = self.get_key(id1, id2)
        if key in self.cache:
            # Update access timestamp
            self.access_timestamps[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, id1: str, id2: str, similarity: float) -> None:
        """Cache similarity value."""
        key = self.get_key(id1, id2)
        
        # If cache is full, remove least recently used item
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_timestamps.items(), key=lambda x: x[1])[0]
            del self.cache[oldest_key]
            del self.access_timestamps[oldest_key]
        
        # Cache new value
        self.cache[key] = similarity
        self.access_timestamps[key] = time.time()


def calculate_o_information(concepts: List[ConceptTuple], k: int = 3) -> Dict[Tuple[str, ...], float]:
    """
    Calculate O-information for sets of k concepts to detect higher-order redundancies.
    
    O-information measures redundancy beyond pairwise comparisons by analyzing
    information relationships among triplets (or more) of concepts.
    
    Args:
        concepts: List of ConceptTuple objects
        k: Size of concept groups to analyze (default: 3 for triplets)
        
    Returns:
        Dictionary mapping k-tuples of concept IDs to O-information values
    """
    if len(concepts) < k:
        return {}
    
    # Create embeddings matrix for efficient computation
    embeddings = np.array([c.embedding for c in concepts if c.embedding is not None])
    concept_ids = [c.eigenfunction_id for c in concepts if c.embedding is not None]
    
    if len(embeddings) < k:
        return {}
    
    # Normalize embeddings for meaningful cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-10)
    
    # Calculate correlation matrix
    corr_matrix = normalized @ normalized.T
    
    # Function to calculate mutual information from correlation
    def mi_from_corr(corr):
        if abs(corr) >= 1.0:
            return 100.0  # Very high value for perfect correlation
        return -0.5 * np.log(1 - corr**2)
    
    # Calculate O-information for all k-tuples
    results = {}
    
    from itertools import combinations
    for indices in combinations(range(len(concept_ids)), k):
        # Get concept IDs for this tuple
        ids_tuple = tuple(concept_ids[i] for i in indices)
        
        # Calculate mutual information for all pairs
        pairwise_mi = []
        for i, j in combinations(range(k), 2):
            idx1, idx2 = indices[i], indices[j]
            corr = corr_matrix[idx1, idx2]
            pairwise_mi.append(mi_from_corr(corr))
        
        # Calculate multi-information for the group
        try:
            # For k=3, O-information = I(X;Y) + I(X;Z) + I(Y;Z) - I(X;Y;Z)
            # We approximate multivariate MI using the correlation determinant
            sub_corr = corr_matrix[np.ix_(indices, indices)]
            det = max(1e-10, np.linalg.det(sub_corr))  # Avoid log(0)
            multi_info = -0.5 * np.log(det)
            
            # Calculate O-information
            o_info = sum(pairwise_mi) - multi_info
            
            # Normalize by the sum of pairwise MI
            sum_mi = sum(pairwise_mi)
            if sum_mi > 0:
                normalized_o_info = o_info / sum_mi
            else:
                normalized_o_info = 0.0
                
            results[ids_tuple] = normalized_o_info
        except np.linalg.LinAlgError:
            # Skip if matrix is singular
            pass
    
    return results


def compute_resonance_decay(
    concept: ConceptTuple,
    current_time: Optional[datetime] = None,
    half_life_days: float = 30.0
) -> float:
    """
    Compute time-based resonance decay for a concept.
    
    Args:
        concept: ConceptTuple to calculate decay for
        current_time: Current datetime (uses now if None)
        half_life_days: Resonance half-life in days
        
    Returns:
        Decayed resonance score between 0 and 1
    """
    if current_time is None:
        current_time = datetime.now()
        
    # Use last_accessed or creation_time from source provenance
    last_time = None
    try:
        if hasattr(concept, 'last_accessed') and concept.last_accessed:
            last_time = concept.last_accessed
        elif 'creation_time' in concept.source_provenance:
            time_str = concept.source_provenance['creation_time']
            # Handle various time string formats
            for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'):
                try:
                    last_time = datetime.strptime(time_str, fmt)
                    break
                except ValueError:
                    continue
    except (AttributeError, KeyError, TypeError):
        pass
    
    if last_time is None:
        # If no time information is available, don't decay resonance
        return concept.resonance_score
    
    # Calculate time difference in days
    time_diff = (current_time - last_time).total_seconds() / (24 * 3600)
    
    # Apply exponential decay: resonance * 2^(-time/half_life)
    decay_factor = 2 ** (-time_diff / half_life_days)
    
    return concept.resonance_score * decay_factor


def calculate_adaptive_entropy_threshold(
    concepts: List[ConceptTuple],
    base_threshold: float = 0.2,
    min_threshold: float = 0.1,
    max_threshold: float = 0.8,
    density_threshold: int = 1000,
    scaling_factor: float = 0.5
) -> float:
    """
    Calculate adaptive entropy threshold based on concept density.
    
    As concept density increases, entropy threshold increases to maintain
    storage efficiency. This helps prevent memory bloat as the system scales.
    
    Args:
        concepts: List of ConceptTuple objects
        base_threshold: Base entropy threshold
        min_threshold: Minimum entropy threshold
        max_threshold: Maximum entropy threshold
        density_threshold: Concepts per dimension threshold
        scaling_factor: Controls threshold scaling rate
        
    Returns:
        Adaptive entropy threshold between min_threshold and max_threshold
    """
    if not concepts:
        return base_threshold
        
    # Get embedding dimension from first valid concept
    dim = None
    for c in concepts:
        if hasattr(c, 'embedding') and c.embedding is not None:
            dim = len(c.embedding)
            break
    
    if dim is None or dim == 0:
        return base_threshold
    
    # Calculate concept density (concepts per dimension)
    density = len(concepts) / dim
    
    # Scale threshold based on density
    # As density increases above threshold, entropy requirement increases
    if density <= density_threshold:
        return base_threshold
    
    # Calculate scaling based on how much density exceeds threshold
    scaling = min(1.0, (density / density_threshold - 1) * scaling_factor)
    
    # Apply scaling between base and max threshold
    threshold = base_threshold + scaling * (max_threshold - base_threshold)
    
    # Ensure threshold is within bounds
    return max(min_threshold, min(max_threshold, threshold))


def enhanced_concept_redundancy(
    concept1: ConceptTuple,
    concept2: ConceptTuple,
    config: MemoryGateConfig,
    similarity_cache: Optional[SimilarityCache] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate enhanced redundancy score between two concepts with detailed metrics.
    
    Args:
        concept1, concept2: ConceptTuple instances to compare
        config: MemoryGateConfig with weights and thresholds
        similarity_cache: Optional cache for similarity calculations
        
    Returns:
        Tuple of (redundancy_score, detailed_metrics)
    """
    metrics = {}
    
    # Check cache first if available
    if similarity_cache:
        cached = similarity_cache.get(concept1.eigenfunction_id, concept2.eigenfunction_id)
        if cached is not None:
            return cached, {}  # No detailed metrics for cached results
    
    # 1. Embedding similarity (vector space)
    if hasattr(concept1, 'embedding') and hasattr(concept2, 'embedding') and \
       concept1.embedding is not None and concept2.embedding is not None:
        emb_sim = cosine_similarity(concept1.embedding, concept2.embedding)
    else:
        emb_sim = 0.0
    metrics["embedding_similarity"] = emb_sim
    
    # 2. Cluster membership similarity
    if hasattr(concept1, 'cluster_members') and hasattr(concept2, 'cluster_members') and \
       concept1.cluster_members and concept2.cluster_members:
        cluster_sim = jaccard_similarity(
            set(concept1.cluster_members), 
            set(concept2.cluster_members)
        )
    else:
        cluster_sim = 0.0
    metrics["cluster_similarity"] = cluster_sim
    
    # 3. Text similarity (concept names)
    # Enhanced to split by multiple word separators
    if hasattr(concept1, 'name') and hasattr(concept2, 'name'):
        import re
        words1 = set(re.split(r'[\s_-]', concept1.name.lower()))
        words2 = set(re.split(r'[\s_-]', concept2.name.lower()))
        
        # Filter out very short words
        words1 = {w for w in words1 if len(w) > 1}
        words2 = {w for w in words2 if len(w) > 1}
        
        text_sim = jaccard_similarity(words1, words2)
    else:
        text_sim = 0.0
    metrics["text_similarity"] = text_sim
    
    # 4. Source provenance similarity
    source_sim = 0.0
    if hasattr(concept1, 'source_provenance') and hasattr(concept2, 'source_provenance'):
        # Check if concepts come from same source document
        src1 = concept1.source_provenance.get('source_id', '')
        src2 = concept2.source_provenance.get('source_id', '')
        if src1 and src2 and src1 == src2:
            source_sim = 1.0
        else:
            # Check document domains
            domain1 = concept1.source_provenance.get('domain', '')
            domain2 = concept2.source_provenance.get('domain', '')
            if domain1 and domain2 and domain1 == domain2:
                source_sim = 0.5
    metrics["source_similarity"] = source_sim
    
    # 5. Spectral lineage similarity
    lineage_sim = 0.0
    if hasattr(concept1, 'spectral_lineage') and hasattr(concept2, 'spectral_lineage') and \
       concept1.spectral_lineage and concept2.spectral_lineage:
        # Extract eigenvalues from lineage
        eigs1 = np.array([complex(real, imag) for real, imag in concept1.spectral_lineage])
        eigs2 = np.array([complex(real, imag) for real, imag in concept2.spectral_lineage])
        
        # Find minimum length for comparison
        min_len = min(len(eigs1), len(eigs2))
        if min_len > 0:
            # Compare dominant eigenvalues
            dom_eigs1 = sorted(eigs1, key=lambda x: abs(x), reverse=True)[:min_len]
            dom_eigs2 = sorted(eigs2, key=lambda x: abs(x), reverse=True)[:min_len]
            
            # Calculate similarity as inverse of normalized distance
            distance = np.mean([abs(e1 - e2) for e1, e2 in zip(dom_eigs1, dom_eigs2)])
            max_val = max(np.max(np.abs(dom_eigs1)), np.max(np.abs(dom_eigs2)))
            if max_val > 0:
                norm_distance = distance / max_val
                lineage_sim = 1.0 / (1.0 + 5.0 * norm_distance)  # Scale for sensitivity
    metrics["lineage_similarity"] = lineage_sim
    
    # Compute weighted redundancy score
    weights = {
        "text_similarity": config.text_similarity_weight,
        "embedding_similarity": config.embedding_similarity_weight,
        "cluster_similarity": config.cluster_similarity_weight,
        "source_similarity": 0.1,  # Small weight for source
        "lineage_similarity": 0.1   # Small weight for lineage
    }
    
    # Normalize weights
    weight_sum = sum(weights.values())
    norm_weights = {k: v/weight_sum for k, v in weights.items()}
    
    # Calculate weighted score
    redundancy = sum(norm_weights[k] * metrics[k] for k in norm_weights)
    
    # Cache result
    if similarity_cache:
        similarity_cache.set(concept1.eigenfunction_id, concept2.eigenfunction_id, redundancy)
    
    return redundancy, metrics


def detect_enhanced_redundancies(
    concepts: List[ConceptTuple],
    config: MemoryGateConfig
) -> Tuple[List[Tuple[int, int, float]], Dict[Tuple[str, ...], float]]:
    """
    Detect redundant concepts with enhanced methods including O-information.
    
    Args:
        concepts: List of ConceptTuple objects
        config: Memory gating configuration
        
    Returns:
        Tuple of (redundant_pairs, redundant_groups)
    """
    if len(concepts) < 2:
        return [], {}
    
    # Initialize cache for similarity calculations
    similarity_cache = SimilarityCache(max_size=config.similarity_cache_size)
    
    # 1. Calculate pairwise redundancies
    redundant_pairs = []
    detailed_metrics = {}
    
    for i in range(len(concepts)):
        for j in range(i+1, len(concepts)):
            redundancy, metrics = enhanced_concept_redundancy(
                concepts[i], concepts[j], config, similarity_cache
            )
            
            if redundancy >= config.redundancy_threshold:
                redundant_pairs.append((i, j, redundancy))
                detailed_metrics[(i, j)] = metrics
                logger.info(f"Redundant concepts detected: '{concepts[i].name}' and '{concepts[j].name}'"
                           f" (score: {redundancy:.2f})")
    
    # 2. Calculate O-information for triplets if enabled
    redundant_groups = {}
    if config.use_o_information and len(concepts) >= 3:
        o_info = calculate_o_information(concepts, k=3)
        
        # Filter for significant O-information values
        redundant_groups = {
            ids: value for ids, value in o_info.items()
            if value >= config.o_information_threshold
        }
        
        # Log significant group redundancies
        for ids, value in redundant_groups.items():
            # Find concept names for logging
            names = []
            for id_str in ids:
                for c in concepts:
                    if c.eigenfunction_id == id_str:
                        names.append(c.name)
                        break
            
            logger.info(f"Higher-order redundancy in concepts: {', '.join(names)} "
                       f"(O-information: {value:.2f})")
    
    # Sort redundant pairs by redundancy score (highest first)
    redundant_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return redundant_pairs, redundant_groups


def enhanced_merge_concepts(
    concepts: List[ConceptTuple],
    redundant_pairs: List[Tuple[int, int, float]],
    redundant_groups: Optional[Dict[Tuple[str, ...], float]] = None
) -> Tuple[List[ConceptTuple], Dict[str, Any]]:
    """
    Enhanced merging of redundant concepts with provenance tracking.
    
    Args:
        concepts: List of ConceptTuple objects
        redundant_pairs: List of (idx1, idx2, score) tuples for redundant pairs
        redundant_groups: Dictionary mapping concept ID tuples to O-information values
        
    Returns:
        Tuple of (merged_concepts, merge_history)
    """
    if not concepts:
        return [], {}
        
    if not redundant_pairs and not redundant_groups:
        return list(concepts), {"merges": 0}
    
    # Create a copy of the concepts list
    result = list(concepts)
    
    # Track indices to merge and remove
    to_remove = set()
    
    # Track merge history
    merge_history = {
        "merges": 0,
        "merge_details": [],
        "group_merges": 0,
        "group_merge_details": []
    }
    
    # 1. First process pairwise redundancies
    for idx1, idx2, redundancy in redundant_pairs:
        # Skip if either concept has already been merged
        if idx1 in to_remove or idx2 in to_remove:
            continue
            
        c1 = result[idx1]
        c2 = result[idx2]
        
        # Create merged concept
        merge_name = f"{c1.name} / {c2.name}"
        
        # Create merged embedding (weighted by resonance)
        w1 = c1.resonance_score 
        w2 = c2.resonance_score
        total_weight = w1 + w2
        
        if total_weight > 0:
            merged_embedding = (w1 * c1.embedding + w2 * c2.embedding) / total_weight
        else:
            merged_embedding = (c1.embedding + c2.embedding) / 2
        
        # Normalize embedding
        norm = np.linalg.norm(merged_embedding)
        if norm > 0:
            merged_embedding = merged_embedding / norm
        
        # Merge other properties
        merged_members = sorted(list(set(c1.cluster_members + c2.cluster_members)))
        merged_context = c1.context if len(c1.context) >= len(c2.context) else c2.context
        
        # Use weighted average for scores
        merged_resonance = max(c1.resonance_score, c2.resonance_score)  # Take maximum resonance
        merged_centrality = (w1 * c1.narrative_centrality + w2 * c2.narrative_centrality) / total_weight if total_weight > 0 else (c1.narrative_centrality + c2.narrative_centrality) / 2
        merged_predictability = (w1 * c1.predictability_score + w2 * c2.predictability_score) / total_weight if total_weight > 0 else (c1.predictability_score + c2.predictability_score) / 2
        
        # Combine spectral lineage data with provenance
        merged_lineage = c1.spectral_lineage + c2.spectral_lineage
        
        # Combine source provenance with merge history
        merged_provenance = {**c1.source_provenance, **c2.source_provenance}
        
        # Add merge record to provenance
        merged_provenance["merge_history"] = merged_provenance.get("merge_history", []) + [{
            "timestamp": datetime.now().isoformat(),
            "merged_concepts": [c1.eigenfunction_id, c2.eigenfunction_id],
            "concept_names": [c1.name, c2.name],
            "redundancy_score": redundancy
        }]
        
        # Create new eigenfunction ID incorporating both parents
        parent_ids = [c1.eigenfunction_id, c2.eigenfunction_id]
        parent_ids.sort()  # For deterministic ID generation
        merged_eigen_id = f"eigen-merged-{hash(''.join(parent_ids)) % 10000000}"
        
        # Calculate merged coherence as maximum of parent coherences
        # since merge should not decrease coherence
        merged_coherence = max(c1.cluster_coherence, c2.cluster_coherence)
        
        # Create passage embedding as combination of both
        if hasattr(c1, 'passage_embedding') and hasattr(c2, 'passage_embedding') and \
           c1.passage_embedding is not None and c2.passage_embedding is not None:
            merged_passage_emb = (c1.passage_embedding + c2.passage_embedding) / 2
        elif hasattr(c1, 'passage_embedding') and c1.passage_embedding is not None:
            merged_passage_emb = c1.passage_embedding
        elif hasattr(c2, 'passage_embedding') and c2.passage_embedding is not None:
            merged_passage_emb = c2.passage_embedding
        else:
            merged_passage_emb = None
        
        # Create merged ConceptTuple
        merged_concept = ConceptTuple(
            name=merge_name,
            embedding=merged_embedding,
            context=merged_context,
            passage_embedding=merged_passage_emb,
            cluster_members=merged_members,
            resonance_score=merged_resonance,
            narrative_centrality=merged_centrality,
            predictability_score=merged_predictability,
            eigenfunction_id=merged_eigen_id,
            source_provenance=merged_provenance,
            spectral_lineage=merged_lineage,
            cluster_coherence=merged_coherence
        )
        
        # Replace the first concept with merged concept
        result[idx1] = merged_concept
        # Mark second concept for removal
        to_remove.add(idx2)
        
        # Add to merge history
        merge_history["merges"] += 1
        merge_history["merge_details"].append({
            "concept1": c1.name,
            "concept2": c2.name,
            "redundancy": redundancy,
            "merged_name": merge_name,
            "eigenfunction_id": merged_eigen_id
        })
        
        logger.info(f"Merged concepts: '{c1.name}' + '{c2.name}' → '{merge_name}'")
    
    # 2. Process group redundancies if available
    if redundant_groups:
        # Map eigenfunction IDs to indices
        id_to_idx = {}
        for i, concept in enumerate(result):
            if i not in to_remove:  # Only consider non-removed concepts
                id_to_idx[concept.eigenfunction_id] = i
        
        # Process each redundant group
        for group_ids, o_info in redundant_groups.items():
            # Find indices of concepts in this group
            group_indices = []
            for id_str in group_ids:
                if id_str in id_to_idx:
                    group_indices.append(id_to_idx[id_str])
            
            # Skip if fewer than 3 concepts available (already merged in pairs)
            if len(group_indices) < 3:
                continue
                
            # Skip if any concept already marked for removal
            if any(idx in to_remove for idx in group_indices):
                continue
            
            # Get concepts
            group_concepts = [result[idx] for idx in group_indices]
            
            # Create group name
            group_name = " / ".join(c.name for c in group_concepts[:3])
            if len(group_concepts) > 3:
                group_name += f" + {len(group_concepts) - 3} more"
            
            # Create merged embedding (weighted by resonance)
            weights = [c.resonance_score for c in group_concepts]
            total_weight = sum(weights)
            
            if total_weight > 0:
                merged_embedding = sum(w * c.embedding for w, c in zip(weights, group_concepts)) / total_weight
            else:
                merged_embedding = sum(c.embedding for c in group_concepts) / len(group_concepts)
            
            # Normalize embedding
            norm = np.linalg.norm(merged_embedding)
            if norm > 0:
                merged_embedding = merged_embedding / norm
            
            # Merge other properties
            all_members = []
            for c in group_concepts:
                all_members.extend(c.cluster_members)
            merged_members = sorted(list(set(all_members)))
            
            # Use longest context
            merged_context = max(
                (c.context for c in group_concepts if hasattr(c, 'context')), 
                key=len, 
                default=""
            )
            
            # Merge scores (maximizing resonance, averaging others)
            merged_resonance = max(c.resonance_score for c in group_concepts)
            if total_weight > 0:
                merged_centrality = sum(w * c.narrative_centrality for w, c in zip(weights, group_concepts)) / total_weight
                merged_predictability = sum(w * c.predictability_score for w, c in zip(weights, group_concepts)) / total_weight
            else:
                merged_centrality = sum(c.narrative_centrality for c in group_concepts) / len(group_concepts)
                merged_predictability = sum(c.predictability_score for c in group_concepts) / len(group_concepts)
            
            # Combine lineage
            merged_lineage = []
            for c in group_concepts:
                merged_lineage.extend(c.spectral_lineage)
            
            # Combine provenance
            merged_provenance = {}
            for c in group_concepts:
                merged_provenance.update(c.source_provenance)
            
            # Add merge record
            merged_provenance["merge_history"] = merged_provenance.get("merge_history", []) + [{
                "timestamp": datetime.now().isoformat(),
                "merged_concepts": [c.eigenfunction_id for c in group_concepts],
                "o_information": o_info,
                "concept_names": [c.name for c in group_concepts],
                "group_name": group_name
            }]

            # Create new eigenfunction ID for group merge
            parent_ids = sorted([c.eigenfunction_id for c in group_concepts])
            merged_eigen_id = f"eigen-group-{hash(''.join(parent_ids)) % 10000000}"

            # Calculate coherence
            merged_coherence = max(c.cluster_coherence for c in group_concepts if hasattr(c, 'cluster_coherence'))

            # Create passage embedding as combination
            passage_embeddings = [
                c.passage_embedding for c in group_concepts
                if hasattr(c, 'passage_embedding') and c.passage_embedding is not None
            ]
            if passage_embeddings:
                merged_passage_emb = sum(passage_embeddings) / len(passage_embeddings)
            else:
                merged_passage_emb = None

            # Create merged ConceptTuple
            merged_concept = ConceptTuple(
                name=group_name,
                embedding=merged_embedding,
                context=merged_context,
                passage_embedding=merged_passage_emb,
                cluster_members=merged_members,
                resonance_score=merged_resonance,
                narrative_centrality=merged_centrality,
                predictability_score=merged_predictability,
                eigenfunction_id=merged_eigen_id,
                source_provenance=merged_provenance,
                spectral_lineage=merged_lineage,
                cluster_coherence=merged_coherence
            )

            # Keep the first concept, mark others for removal
            result[group_indices[0]] = merged_concept
            for idx in group_indices[1:]:
                to_remove.add(idx)

            # Add to merge history
            merge_history["group_merges"] += 1
            merge_history["group_merge_details"].append({
                "concepts": [c.name for c in group_concepts],
                "o_information": o_info,
                "merged_name": group_name,
                "eigenfunction_id": merged_eigen_id
            })

            logger.info(f"Merged group concepts: {[c.name for c in group_concepts]} → '{group_name}'")

    # Remove merged concepts (in reverse order to preserve indices)
    for idx in sorted(to_remove, reverse=True):
        result.pop(idx)

    return result, merge_history


def enhanced_prune_low_entropy_concepts(
    concepts: List[ConceptTuple],
    config: MemoryGateConfig,
    current_time: Optional[datetime] = None,
) -> Tuple[List[ConceptTuple], Dict[str, Any]]:
    """
    Enhanced pruning of low-entropy or low-resonance concepts.
    
    Implementation of the "No Memory Bloat" commitment using spectral entropy
    and time-based resonance decay to identify concepts for pruning.
    
    Args:
        concepts: List of ConceptTuple objects
        config: Memory gating configuration
        current_time: Current datetime for resonance decay calculation
        
    Returns:
        Tuple of (pruned_concepts, pruning_stats)
    """
    if not concepts:
        return [], {"pruned_count": 0}
    
    # Calculate adaptive entropy threshold
    if config.adaptive_entropy_scaling:
        entropy_threshold = calculate_adaptive_entropy_threshold(
            concepts,
            base_threshold=config.base_entropy_threshold,
            min_threshold=config.min_entropy_threshold,
            max_threshold=config.max_entropy_threshold,
            density_threshold=config.concept_density_threshold,
            scaling_factor=config.density_scaling_factor
        )
    else:
        entropy_threshold = config.base_entropy_threshold
    
    # Track concept pruning
    pruning_stats = {
        "initial_count": len(concepts),
        "entropy_threshold": entropy_threshold,
        "using_resonance_decay": config.enable_resonance_decay,
        "pruned_count": 0,
        "pruned_by_entropy": 0,
        "pruned_by_resonance": 0,
        "pruned_concepts": []
    }
    
    result = []
    
    # Calculate entropy for each concept
    concept_entropies = {}
    for concept in concepts:
        embedding = concept.embedding
        if embedding is not None and len(embedding) > 0:
            entropy = calculate_spectral_entropy(embedding)
            concept_entropies[concept.eigenfunction_id] = entropy
    
    # Process each concept
    for concept in concepts:
        # Skip concepts with missing eigenfunction_id
        if not hasattr(concept, 'eigenfunction_id') or not concept.eigenfunction_id:
            result.append(concept)
            continue
        
        # Check entropy
        entropy = concept_entropies.get(concept.eigenfunction_id, 1.0)
        prune_by_entropy = entropy < entropy_threshold
        
        # Check resonance decay if enabled
        prune_by_resonance = False
        if config.enable_resonance_decay:
            # Calculate decayed resonance
            decayed_resonance = compute_resonance_decay(
                concept, current_time, config.half_life_days
            )
            
            # Prune if resonance below minimum threshold
            prune_by_resonance = decayed_resonance < config.min_resonance_keep
        
        # Prune if either condition is met
        if prune_by_entropy or prune_by_resonance:
            # Add to pruning stats
            if prune_by_entropy:
                pruning_stats["pruned_by_entropy"] += 1
            if prune_by_resonance:
                pruning_stats["pruned_by_resonance"] += 1
                
            pruning_stats["pruned_count"] += 1
            pruning_stats["pruned_concepts"].append({
                "name": concept.name,
                "eigenfunction_id": concept.eigenfunction_id,
                "entropy": entropy,
                "resonance": getattr(concept, 'resonance_score', None),
                "by_entropy": prune_by_entropy,
                "by_resonance": prune_by_resonance
            })
            
            logger.info(f"Pruned concept: '{concept.name}' "
                        f"(entropy: {entropy:.3f}, threshold: {entropy_threshold:.3f}, "
                        f"by_resonance: {prune_by_resonance})")
        else:
            # Keep this concept
            result.append(concept)
    
    pruning_stats["final_count"] = len(result)
    pruning_stats["reduction_percent"] = (
        (pruning_stats["initial_count"] - pruning_stats["final_count"]) / 
        max(1, pruning_stats["initial_count"]) * 100
    )
    
    logger.info(f"Pruned {pruning_stats['pruned_count']} concepts "
                f"({pruning_stats['reduction_percent']:.1f}% reduction)")
    
    return result, pruning_stats


def apply_enhanced_memory_gating(
    concepts: List[ConceptTuple],
    config: Optional[MemoryGateConfig] = None,
    current_time: Optional[datetime] = None
) -> Tuple[List[ConceptTuple], Dict[str, Any]]:
    """
    Apply enhanced memory gating to prevent concept store bloat.
    
    This is the main function implementing ALAN's "No Memory Bloat" commitment.
    It performs a sequence of operations:
    1. Prune low-entropy concepts that don't contribute unique information
    2. Apply time-based resonance decay to remove stale concepts
    3. Detect and merge redundant concepts
    
    Args:
        concepts: List of ConceptTuple objects to process
        config: Memory gating configuration (uses defaults if None)
        current_time: Current datetime for time-based operations
        
    Returns:
        Tuple of (processed_concepts, stats)
    """
    if not concepts:
        return [], {"status": "no_concepts"}
    
    # Use default config if not provided
    if config is None:
        config = MemoryGateConfig()
    
    # Initialize stats tracking
    stats = {
        "initial_count": len(concepts),
        "entropy_threshold": 0.0,
        "config": config.to_dict(),
        "timestamp": datetime.now().isoformat()
    }
    
    # 1. Prune low-entropy and decayed concepts
    pruned_concepts, pruning_stats = enhanced_prune_low_entropy_concepts(
        concepts, config, current_time
    )
    stats["pruning"] = pruning_stats
    
    # Skip further processing if too few concepts remain
    if len(pruned_concepts) < 2:
        stats["final_count"] = len(pruned_concepts)
        stats["status"] = "completed_after_pruning"
        return pruned_concepts, stats
    
    # 2. Detect redundancies
    redundant_pairs, redundant_groups = detect_enhanced_redundancies(
        pruned_concepts, config
    )
    
    stats["redundancy"] = {
        "redundant_pairs": len(redundant_pairs),
        "redundant_groups": len(redundant_groups)
    }
    
    # 3. Merge redundant concepts
    if redundant_pairs or redundant_groups:
        merged_concepts, merge_history = enhanced_merge_concepts(
            pruned_concepts, redundant_pairs, redundant_groups
        )
        stats["merging"] = merge_history
    else:
        merged_concepts = pruned_concepts
        stats["merging"] = {"merges": 0, "group_merges": 0}
    
    # Finalize stats
    stats["final_count"] = len(merged_concepts)
    stats["reduction_percent"] = (
        (stats["initial_count"] - stats["final_count"]) / 
        max(1, stats["initial_count"]) * 100
    )
    stats["status"] = "completed"
    
    logger.info(f"Memory gating complete: {stats['initial_count']} → {stats['final_count']} concepts "
                f"({stats['reduction_percent']:.1f}% reduction)")
    
    return merged_concepts, stats

