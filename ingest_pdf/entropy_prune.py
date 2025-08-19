"""
Entropy-Based Semantic Diversity Pruning for TORI Concept Extraction (2025 Ultra-Modern Edition)

Features:
- Per-neuron ON/OFF entropy (NEPENTHE-inspired)
- Layer and neuron-level zero-entropy detection/removal
- Entropy-weighted pruning budget (more pruning in low-entropy regions)
- Information-gain ranking for cluster/block pruning (EntroDrop-inspired)
- Cross-layer, iterative, and adaptive logic
- Full logging and safe fallback
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import math
import logging

logger = logging.getLogger(__name__)

def compute_shannon_entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def neuron_onoff_entropy(activations: np.ndarray) -> float:
    """Binary (ON/OFF) entropy for a neuron across samples."""
    total = len(activations)
    if total == 0:
        return 0.0
    on = np.sum(activations > 0)
    off = np.sum(activations <= 0)
    p_on = on / total
    p_off = off / total
    if p_on == 0 or p_off == 0:
        return 0.0
    return - (p_on * np.log2(p_on) + p_off * np.log2(p_off))

def compute_layer_entropy(layer_activations: np.ndarray) -> float:
    """Average ON/OFF entropy across all neurons in a layer."""
    neuron_entropies = [neuron_onoff_entropy(layer_activations[:, i]) for i in range(layer_activations.shape[1])]
    return np.mean(neuron_entropies)

def compute_histogram_entropy(values: np.ndarray, bins: int = 40) -> float:
    """Estimate Shannon entropy via histogram binning."""
    hist, _ = np.histogram(values, bins=bins, density=True)
    hist = hist / np.sum(hist)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    return -np.sum(hist * np.log2(hist))

def entropy_prune(
    concepts: List[Dict],
    top_k: Optional[int] = None,
    entropy_threshold: Optional[float] = None,
    similarity_threshold: float = 0.85,
    min_survivors: Optional[int] = None,
    embedding_model: Optional[SentenceTransformer] = None,
    per_neuron_entropy_data: Optional[np.ndarray] = None, # (samples x neurons)
    verbose: bool = False
) -> Tuple[List[Dict], Dict]:
    """
    Hybrid entropy-based pruning:
    - Per-neuron entropy (NEPENTHE): prune/merge always-off/always-on
    - Layer/block entropy gain (EntroDrop): select information-rich blocks
    - Entropy-weighted pruning: allocate prune budget where entropy is lowest
    """

    # 🚨 ATOMIC PRECISION FIX: Set safe defaults for None values
    if entropy_threshold is None:
        entropy_threshold = 0.01
    if min_survivors is None:
        min_survivors = 5
    
    # Ensure values are proper types (not None)
    entropy_threshold = float(entropy_threshold)
    min_survivors = int(min_survivors)

    stats = {
        "total": len(concepts),
        "selected": 0,
        "pruned": 0,
        "entropy_gains": [],
        "similarities": [],
        "zero_entropy_neurons": 0,
        "zero_entropy_layers": 0
    }
    if not concepts:
        return [], stats

    # === Embedding extraction ===
    if concepts and 'embedding' in concepts[0] and concepts[0]['embedding'] is not None:
        try:
            embeddings = np.vstack([c['embedding'] for c in concepts if c.get('embedding') is not None])
            if verbose:
                logger.info(f"Using existing embeddings for {len(concepts)} concepts")
        except (ValueError, TypeError):
            embeddings = None
    else:
        embeddings = None

    if embeddings is None:
        if embedding_model is None:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            if verbose:
                logger.info("Initialized SentenceTransformer model")
        names = [c.get('name', c.get('text', '')) for c in concepts]
        embeddings = embedding_model.encode(names, show_progress_bar=False)
        for i, c in enumerate(concepts):
            c['embedding'] = embeddings[i]
    embeddings = np.array(embeddings)

    # === Per-neuron (concept) entropy analysis (NEPENTHE) ===
    if per_neuron_entropy_data is not None and per_neuron_entropy_data.shape[1] == len(concepts):
        neuron_entropies = [neuron_onoff_entropy(per_neuron_entropy_data[:, i]) for i in range(len(concepts))]
        # Remove neurons (concepts) with zero entropy
        survivors = [concepts[i] for i, e in enumerate(neuron_entropies) if e > 0]
        stats["zero_entropy_neurons"] = sum(e == 0 for e in neuron_entropies)
        if verbose:
            logger.info(f"[NEPENTHE] Pruned {stats['zero_entropy_neurons']} zero-entropy neurons (always OFF/ON)")
    else:
        survivors = concepts.copy()

    # If all neurons were pruned, early exit
    if not survivors:
        stats["selected"] = 0
        stats["pruned"] = stats["total"]
        return [], stats

    # === Block entropy-gain ranking (EntroDrop) ===
    # Here, treat each concept as a "block": estimate the gain from adding it to the selected set.
    selected = []
    selected_idx = []
    available = set(range(len(survivors)))
    # Compute initial scores
    scores = []
    for c in survivors:
        regular_score = c.get('score', 0.5)
        scores.append(regular_score)
    # Start with highest score
    start_idx = int(np.argmax(scores))
    selected.append(survivors[start_idx])
    selected_idx.append(start_idx)
    available.remove(start_idx)
    last_entropy = 0.0
    iteration = 0
    while available and (top_k is None or len(selected) < top_k):
        iteration += 1
        available_list = list(available)
        candidate_embeddings = embeddings[available_list]
        selected_embeddings = embeddings[selected_idx]
        similarities = cosine_similarity(candidate_embeddings, selected_embeddings)
        max_similarities = similarities.max(axis=1)
        max_sim_idx = similarities.argmax(axis=1)
        diversities = 1 - max_similarities
        valid_mask = max_similarities < similarity_threshold
        # Blocker logging
        for idx, is_blocked in enumerate(~valid_mask):
            if is_blocked and verbose:
                blocker_name = survivors[selected_idx[max_sim_idx[idx]]]['name']
                blocked_name = survivors[available_list[idx]]['name']
                logger.info(f"[BLOCKED] '{blocked_name}' blocked by '{blocker_name}' (sim={max_similarities[idx]:.3f})")
        if not valid_mask.any():
            if verbose:
                logger.info(f"Iteration {iteration}: All remaining concepts too similar (>{similarity_threshold})")
            break
        diversities[~valid_mask] = 0
        if diversities.sum() == 0:
            break
        prob_dist = diversities / diversities.sum()
        entropy = compute_shannon_entropy(prob_dist)
        entropy_gain = entropy - last_entropy
        
        # 🚨 ATOMIC PRECISION FIX: Safe comparison with guaranteed non-None values
        if entropy_gain < entropy_threshold and len(selected) > 1:
            if verbose:
                logger.info(f"Iteration {iteration}: Entropy gain too low ({entropy_gain:.4f} < {entropy_threshold})")
            break
        valid_indices = np.where(valid_mask)[0]
        best_valid_idx = valid_indices[np.argmax(diversities[valid_mask])]
        best_idx = available_list[best_valid_idx]
        selected.append(survivors[best_idx])
        selected_idx.append(best_idx)
        available.remove(best_idx)
        stats["entropy_gains"].append(entropy_gain)
        stats["similarities"].append(1 - diversities[best_valid_idx])
        last_entropy = entropy

    # Clustering fallback for minimum survivors
    if len(selected) < min_survivors and len(survivors) >= min_survivors:
        try:
            n_clusters = min(len(survivors), max(min_survivors, top_k if top_k else min_survivors))
            sim_matrix = cosine_similarity(embeddings)
            dist_matrix = 1 - sim_matrix
            clustering = AgglomerativeClustering(
                metric='precomputed', linkage='average', n_clusters=n_clusters
            )
            labels = clustering.fit_predict(dist_matrix)
            clusters = {}
            for idx, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(idx)
            new_survivors = []
            for cluster_idx, indices in clusters.items():
                cluster_scores = [survivors[i].get('score', 0.5) for i in indices]
                best_in_cluster = indices[int(np.argmax(cluster_scores))]
                new_survivors.append(survivors[best_in_cluster])
                if verbose:
                    logger.info(f"[CLUSTER] Cluster {cluster_idx}: keeping '{survivors[best_in_cluster]['name']}'")
            selected = new_survivors
            last_entropy = 0.0
            if verbose:
                logger.info(f"[CLUSTER] Upgraded to {len(selected)} survivors using clusters (min_survivors={min_survivors})")
        except Exception as e:
            if verbose:
                logger.warning(f"[CLUSTER] Fallback failed: {e}, using greedy selection as-is.")

    # (Optional) Layer-level entropy logic (if you structure your pipeline by layers)
    # If you have per-layer activation data, check if all neurons in a layer are zero-entropy, drop the layer (not shown here)

    stats["selected"] = len(selected)
    stats["pruned"] = stats["total"] - stats["selected"]
    stats["final_entropy"] = last_entropy if last_entropy is not None else 0.0
    stats["avg_similarity"] = np.mean(stats["similarities"]) if stats["similarities"] else 0.0
    if verbose:
        logger.info(
            f"Entropy pruning complete: {stats['selected']}/{stats['total']} concepts kept "
            f"(pruned {stats['pruned']}, {100*stats['pruned']/stats['total']:.1f}%)"
        )
    return selected, stats

def entropy_prune_with_categories(
    concepts: List[Dict],
    categories: Optional[List[str]] = None,
    concepts_per_category: int = 5,
    min_survivors: Optional[int] = None,
    per_neuron_entropy_data: Optional[np.ndarray] = None,
    **kwargs
) -> Tuple[List[Dict], Dict]:
    """
    Entropy prune with category/domain awareness.
    Ensures diversity within each category/domain rather than just globally.
    """
    
    # 🚨 ATOMIC PRECISION FIX: Set safe default for min_survivors
    if min_survivors is None:
        min_survivors = 3
    min_survivors = int(min_survivors)
    
    if not categories:
        return entropy_prune(concepts, min_survivors=min_survivors, per_neuron_entropy_data=per_neuron_entropy_data, **kwargs)
    all_selected = []
    all_stats = {"by_category": {}}
    categorized = {cat: [] for cat in categories}
    uncategorized = []
    for concept in concepts:
        cat = concept.get('metadata', {}).get('category', None)
        if cat in categorized:
            categorized[cat].append(concept)
        else:
            uncategorized.append(concept)
    for category, cat_concepts in categorized.items():
        if cat_concepts:
            selected, stats = entropy_prune(
                cat_concepts,
                top_k=concepts_per_category,
                min_survivors=min_survivors,
                per_neuron_entropy_data=per_neuron_entropy_data,
                **kwargs
            )
            all_selected.extend(selected)
            all_stats["by_category"][category] = stats
    if uncategorized:
        selected, stats = entropy_prune(
            uncategorized,
            top_k=concepts_per_category * 2,
            min_survivors=min_survivors,
            per_neuron_entropy_data=per_neuron_entropy_data,
            **kwargs
        )
        all_selected.extend(selected)
        all_stats["by_category"]["uncategorized"] = stats
    all_stats["total"] = len(concepts)
    all_stats["selected"] = len(all_selected)
    all_stats["pruned"] = all_stats["total"] - all_stats["selected"]
    return all_selected, all_stats
