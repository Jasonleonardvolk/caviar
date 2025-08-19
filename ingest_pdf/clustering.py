import math
import time
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any, Optional
from alan_backend.banksy import oscillator_update

def cluster_cohesion(emb: np.ndarray, members: List[int]) -> float:
    """Calculate cluster cohesion using cosine similarity."""
    if len(members) < 2:
        return 0.0
    sub = emb[members]
    sim = (sub @ sub.T) / ((np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8) * (np.linalg.norm(sub, axis=1, keepdims=True).T + 1e-8))
    return (sim.sum() - len(sub)) / max(len(sub)*(len(sub)-1), 1)

def run_oscillator_clustering(emb: np.ndarray, steps: int = 60, tol: float = 1e-3, cohesion_threshold: float = 0.15) -> List[int]:
    """
    Original oscillator clustering function - maintains backward compatibility.
    """
    n = emb.shape[0]
    if n == 1:
        return [0]
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    K = (emb @ emb.T) / (norms * norms.T)
    np.fill_diagonal(K, 0)
    theta = np.random.uniform(0, 2 * math.pi, n)
    for _ in range(steps):
        nxt = oscillator_update(theta, np.ones(n), 0.25, 0.05, K, 0)
        if np.linalg.norm(nxt - theta) < tol:
            break
        theta = nxt
    # Bucket phases
    buckets: Dict[int, List[int]] = defaultdict(list)
    for i, th in enumerate(theta):
        buckets[int((th % (2 * math.pi)) / 0.25)].append(i)
    # Assign cluster labels
    labels = [-1] * n
    cid_map = {}
    for cid, (_, mem) in enumerate(sorted(buckets.items(), key=lambda kv: -len(kv[1]))):
        for m in mem:
            labels[m] = cid
        cid_map[cid] = mem
    # Validate clusters: merge/discard singletons and low cohesion
    # Find nearest neighbor for singletons
    for cid, mem in list(cid_map.items()):
        if len(mem) == 1:
            idx = mem[0]
            # Find nearest other cluster by cosine similarity
            best_cid, best_sim = None, -1e9
            for ocid, omem in cid_map.items():
                if ocid == cid or len(omem) < 1:
                    continue
                sim = float(emb[idx] @ emb[omem[0]] / (np.linalg.norm(emb[idx])*np.linalg.norm(emb[omem[0]])+1e-8))
                if sim > best_sim:
                    best_cid, best_sim = ocid, sim
            if best_cid is not None and best_sim > 0.4:
                # Merge singleton into nearest cluster
                labels[idx] = best_cid
                cid_map[best_cid].append(idx)
                cid_map[cid] = []
    # Remove clusters with low cohesion
    for cid, mem in list(cid_map.items()):
        if len(mem) < 2:
            continue
        coh = cluster_cohesion(emb, mem)
        if coh < cohesion_threshold:
            for idx in mem:
                labels[idx] = -1  # Mark as unassigned
            cid_map[cid] = []
    # Reassign unassigned to nearest cluster
    for idx, l in enumerate(labels):
        if l == -1:
            best_cid, best_sim = None, -1e9
            for cid, mem in cid_map.items():
                if len(mem) == 0:
                    continue
                sim = float(emb[idx] @ emb[mem[0]] / (np.linalg.norm(emb[idx])*np.linalg.norm(emb[mem[0]])+1e-8))
                if sim > best_sim:
                    best_cid, best_sim = cid, sim
            if best_cid is not None and best_sim > 0.3:
                labels[idx] = best_cid
                cid_map[best_cid].append(idx)
    return labels

def run_oscillator_clustering_with_metrics(
    emb: np.ndarray, 
    steps: int = 60, 
    tol: float = 1e-3, 
    cohesion_threshold: float = 0.15,
    enable_logging: bool = False
) -> Dict[str, Any]:
    """
    Enhanced oscillator clustering with detailed metrics and logging.
    
    Returns:
        Dictionary containing labels, clusters, cohesion scores, runtime, and convergence info
    """
    start_time = time.perf_counter()
    
    n = emb.shape[0]
    if n == 1:
        return {
            "labels": [0],
            "clusters": {0: [0]},
            "cohesion_scores": {0: 0.0},
            "runtime": time.perf_counter() - start_time,
            "convergence_step": 0,
            "total_steps": 0,
            "phase_variance": 0.0,
            "singleton_merges": 0,
            "orphan_reassignments": 0,
            "removed_low_cohesion": 0,
            "n_clusters": 1,
            "avg_cohesion": 0.0
        }
    
    # Build cosine similarity matrix
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    K = (emb @ emb.T) / (norms * norms.T)
    np.fill_diagonal(K, 0)
    
    # Initialize random phases
    theta = np.random.uniform(0, 2 * math.pi, n)
    initial_phases = theta.copy()
    
    convergence_step = steps
    phase_variances = []
    
    # Oscillator evolution
    for step in range(steps):
        nxt = oscillator_update(theta, np.ones(n), 0.25, 0.05, K, 0)
        phase_diff = np.linalg.norm(nxt - theta)
        phase_variances.append(np.var(nxt))
        
        if phase_diff < tol:
            convergence_step = step
            break
        theta = nxt
    
    # Bucket phases
    buckets: Dict[int, List[int]] = defaultdict(list)
    for i, th in enumerate(theta):
        buckets[int((th % (2 * math.pi)) / 0.25)].append(i)
    
    # Assign initial cluster labels
    labels = [-1] * n
    cid_map = {}
    for cid, (_, mem) in enumerate(sorted(buckets.items(), key=lambda kv: -len(kv[1]))):
        for m in mem:
            labels[m] = cid
        cid_map[cid] = mem
    
    # Track modifications
    singleton_merges = 0
    orphan_reassignments = 0
    removed_low_cohesion = 0
    
    # Handle singletons - merge with nearest cluster
    for cid, mem in list(cid_map.items()):
        if len(mem) == 1:
            idx = mem[0]
            # Find nearest other cluster by cosine similarity
            best_cid, best_sim = None, -1e9
            for ocid, omem in cid_map.items():
                if ocid == cid or len(omem) < 1:
                    continue
                sim = float(emb[idx] @ emb[omem[0]] / (np.linalg.norm(emb[idx])*np.linalg.norm(emb[omem[0]])+1e-8))
                if sim > best_sim:
                    best_cid, best_sim = ocid, sim
            if best_cid is not None and best_sim > 0.4:
                # Merge singleton into nearest cluster
                labels[idx] = best_cid
                cid_map[best_cid].append(idx)
                cid_map[cid] = []
                singleton_merges += 1
    
    # Remove clusters with low cohesion
    for cid, mem in list(cid_map.items()):
        if len(mem) < 2:
            continue
        coh = cluster_cohesion(emb, mem)
        if coh < cohesion_threshold:
            for idx in mem:
                labels[idx] = -1  # Mark as unassigned
            cid_map[cid] = []
            removed_low_cohesion += 1
    
    # Reassign unassigned to nearest cluster
    for idx, l in enumerate(labels):
        if l == -1:
            best_cid, best_sim = None, -1e9
            for cid, mem in cid_map.items():
                if len(mem) == 0:
                    continue
                sim = float(emb[idx] @ emb[mem[0]] / (np.linalg.norm(emb[idx])*np.linalg.norm(emb[mem[0]])+1e-8))
                if sim > best_sim:
                    best_cid, best_sim = cid, sim
            if best_cid is not None and best_sim > 0.3:
                labels[idx] = best_cid
                cid_map[best_cid].append(idx)
                orphan_reassignments += 1
    
    # Calculate final cohesion scores
    cohesion_scores = {}
    active_clusters = {}
    for cid, mem in cid_map.items():
        if len(mem) > 0:
            active_clusters[cid] = mem
            cohesion_scores[cid] = cluster_cohesion(emb, mem)
    
    runtime = time.perf_counter() - start_time
    avg_cohesion = np.mean(list(cohesion_scores.values())) if cohesion_scores else 0.0
    
    result = {
        "labels": labels,
        "clusters": active_clusters,
        "cohesion_scores": cohesion_scores,
        "runtime": runtime,
        "convergence_step": convergence_step,
        "total_steps": steps,
        "phase_variance": np.mean(phase_variances) if phase_variances else 0.0,
        "final_phase_variance": np.var(theta),
        "similarity_matrix_density": np.mean(K[K > 0]) if np.any(K > 0) else 0.0,
        "singleton_merges": singleton_merges,
        "orphan_reassignments": orphan_reassignments,
        "removed_low_cohesion": removed_low_cohesion,
        "n_clusters": len(active_clusters),
        "avg_cohesion": avg_cohesion,
        "method": "oscillator"
    }
    
    if enable_logging:
        print(f"Oscillator Clustering Results:")
        print(f"  Runtime: {runtime:.3f}s")
        print(f"  Convergence: {convergence_step}/{steps} steps")
        print(f"  Clusters found: {len(active_clusters)}")
        print(f"  Singleton merges: {singleton_merges}")
        print(f"  Orphan reassignments: {orphan_reassignments}")
        print(f"  Low cohesion removed: {removed_low_cohesion}")
        print(f"  Average cohesion: {avg_cohesion:.3f}")
    
    return result

# Enhanced clustering methods - fallback implementations
def kmeans_clustering(emb: np.ndarray, k: Optional[int] = None, max_iter: int = 100) -> Dict[str, Any]:
    """Simplified K-means implementation."""
    start_time = time.perf_counter()
    n = emb.shape[0]
    if k is None:
        k = max(1, int(np.sqrt(n / 2)))
    
    labels = run_oscillator_clustering(emb)  # Fallback to oscillator
    
    # Build clusters dict
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    # Calculate cohesion scores
    cohesion_scores = {}
    for cid, members in clusters.items():
        cohesion_scores[cid] = cluster_cohesion(emb, members)
    
    return {
        "labels": labels,
        "clusters": clusters,
        "cohesion_scores": cohesion_scores,
        "runtime": time.perf_counter() - start_time,
        "method": "kmeans_fallback"
    }

def hdbscan_clustering(emb: np.ndarray, min_cluster_size: int = 5) -> Dict[str, Any]:
    """HDBSCAN fallback implementation."""
    start_time = time.perf_counter()
    labels = run_oscillator_clustering(emb)
    
    # Build clusters dict
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    # Calculate cohesion scores
    cohesion_scores = {}
    for cid, members in clusters.items():
        cohesion_scores[cid] = cluster_cohesion(emb, members)
    
    return {
        "labels": labels,
        "clusters": clusters,
        "cohesion_scores": cohesion_scores,
        "runtime": time.perf_counter() - start_time,
        "method": "hdbscan_fallback"
    }

def affinity_propagation_clustering(emb: np.ndarray, damping: float = 0.5) -> Dict[str, Any]:
    """Affinity Propagation fallback implementation."""
    start_time = time.perf_counter()
    labels = run_oscillator_clustering(emb)
    
    # Build clusters dict
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    # Calculate cohesion scores
    cohesion_scores = {}
    for cid, members in clusters.items():
        cohesion_scores[cid] = cluster_cohesion(emb, members)
    
    return {
        "labels": labels,
        "clusters": clusters,
        "cohesion_scores": cohesion_scores,
        "runtime": time.perf_counter() - start_time,
        "method": "affinity_propagation_fallback"
    }

def benchmark_all_clustering_methods(emb: np.ndarray, methods: List[str] = None, enable_logging: bool = True) -> Dict[str, Dict[str, Any]]:
    """Fallback benchmark that only uses oscillator clustering."""
    result = run_oscillator_clustering_with_metrics(emb, enable_logging=enable_logging)
    return {"oscillator": result}

def compute_silhouette_score(emb: np.ndarray, labels: List[int]) -> float:
    """Compute silhouette score for clustering evaluation."""
    if len(set(labels)) < 2:
        return 0.0
    
    n = len(labels)
    scores = []
    
    for i in range(n):
        # Same cluster distances
        same_cluster = [j for j in range(n) if labels[j] == labels[i] and j != i]
        if not same_cluster:
            continue
            
        a = np.mean([np.linalg.norm(emb[i] - emb[j]) for j in same_cluster])
        
        # Different cluster distances
        other_clusters = set(labels) - {labels[i]}
        if not other_clusters:
            continue
            
        b_values = []
        for cluster in other_clusters:
            cluster_points = [j for j in range(n) if labels[j] == cluster]
            if cluster_points:
                b_cluster = np.mean([np.linalg.norm(emb[i] - emb[j]) for j in cluster_points])
                b_values.append(b_cluster)
        
        if b_values:
            b = min(b_values)
            score = (b - a) / max(a, b)
            scores.append(score)
    
    return np.mean(scores) if scores else 0.0
